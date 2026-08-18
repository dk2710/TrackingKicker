
import cv2
import copy
import os
import signal
import subprocess
import shutil
import sys
import time
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Dict, Optional
from flask import Flask, jsonify, send_from_directory, url_for

import numpy as np
import traceback

try:
    import fcntl
except ImportError:
    fcntl = None

from .config import AppConfig
from .RawReplaySource import RawReplaySource
from .BaslerSource import BaslerSource
from .calibration import CalibrationPipeline
from .detection import BallDetector
from .analysis import KickerAnalyzer
from .distoration import DistortionCorrector
from .model.VisionConfig import VisionConfig
from .storage import JsonStore
from .ui.distortion_tuner import DistortionTunerUI


# -------------------------------------------------
# Globale Konfiguration
# -------------------------------------------------
vision = VisionConfig.load()
config = AppConfig.load()

VIDEO_PATH = config.videos_dir / "test.npz"

Analyse_Video_Path = (Path(__file__).resolve().parents[1] / "videos").resolve()
Analyse_Video_Path.mkdir(parents=True, exist_ok=True)

# Alte, vom frueheren Segment-Encoder erzeugte Dateien sind keine gueltigen
# aktuellen Goal-Videos mehr. Beim Start werden nur diese automatisch
# erzeugten Artefakte entfernt; goal_latest.webm bleibt erhalten.
for _old_goal_file in Analyse_Video_Path.iterdir():
    if (
        _old_goal_file.is_file()
        and _old_goal_file.name != "goal_latest.webm"
        and (
            _old_goal_file.name.startswith("goal_ring_")
            or _old_goal_file.name.startswith("goal_")
        )
        and _old_goal_file.suffix in {".webm", ".csv"}
    ):
        try:
            _old_goal_file.unlink()
        except OSError as _cleanup_error:
            print(f"[GOAL] Alte Videodatei nicht entfernbar: {_old_goal_file}: {_cleanup_error}")

# Auf Geraeten ohne Monitor darf OpenCV keine HighGUI-Funktionen aufrufen.
# Die Einstellung aus config.json kann per HEADLESS=0 bewusst ueberschrieben
# werden, sofern ein Display vorhanden ist.
_headless_env = os.environ.get("HEADLESS")
_no_linux_display = (
    os.name == "posix"
    and not os.environ.get("DISPLAY")
    and not os.environ.get("WAYLAND_DISPLAY")
)
HEADLESS = (
    config.headless
    if _headless_env is None
    else _headless_env == "1"
) or _no_linux_display
GOAL_VIDEO_DURATION_SECONDS = 3
GOAL_VIDEO_FPS = 60
GOAL_VIDEO_FRAME_LIMIT = GOAL_VIDEO_DURATION_SECONDS * GOAL_VIDEO_FPS
GOAL_VIDEO_MAX_WIDTH = 640

_app_lock_handle = None


def _acquire_app_lock() -> bool:
    """Verhindert, dass zwei Prozesse dieselbe Basler gleichzeitig oeffnen."""
    global _app_lock_handle
    if fcntl is None:
        return True
    try:
        _app_lock_handle = open("/tmp/tracking_kicker.lock", "w")
        fcntl.flock(_app_lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _app_lock_handle.write(str(os.getpid()))
        _app_lock_handle.flush()
        return True
    except (BlockingIOError, OSError):
        print("[APP] Bereits eine TrackingKicker-Instanz aktiv; zweiter Start beendet.")
        return False


# -------------------------------------------------
# GUI helpers
# -------------------------------------------------
def gui_named_window(name: str):
    if not HEADLESS:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)


def gui_imshow(name: str, frame):
    if not HEADLESS:
        cv2.imshow(name, frame)


def gui_wait_key(ms: int = 1) -> int:
    if not HEADLESS:
        return cv2.waitKey(ms) & 0xFF

    return 255


def gui_destroy_window(name: str):
    if not HEADLESS:
        cv2.destroyWindow(name)


def gui_destroy_all():
    if not HEADLESS:
        cv2.destroyAllWindows()


# -------------------------------------------------
# KickerApp
# -------------------------------------------------
class KickerApp:
    def __init__(self, source):
        self.video = source

        self.cfg: Optional[Dict[str, Any]] = None
        self.dist = DistortionCorrector(config.camera)

        print(f"📁 Verzerrung config: {config.camera}")

        cv2.setUseOptimized(True)

        # Queues gehören jetzt zur App
        self.frame_queue = Queue(maxsize=2)
        self.display_queue = Queue(maxsize=1)
        self.recording_queue = Queue()

        self.raw_queue = Queue(maxsize=2)
        self.processed_queue = Queue(maxsize=2)

        self.history_lock = threading.Lock()
        self.goal_video_lock = threading.Lock()
        self.goal_video_ready = threading.Event()
        self.goal_video_path: Optional[str] = None
        self.goal_video_speed = 0.0
        self.goal_video_frame_count = 0
        self.goal_video_fps = GOAL_VIDEO_FPS
        self.goal_video_thread: Optional[threading.Thread] = None
        self.rolling_video_queue = Queue(maxsize=1)
        self.rolling_video_lock = threading.Lock()
        self.rolling_video_path: Optional[str] = None
        self.rolling_video_list: Optional[Path] = None
        self.rolling_video_ready_time = 0.0
        self.rolling_video_thread: Optional[threading.Thread] = None
        self.rolling_last_submit = 0.0
        self.t_api = None
        self.t_cam = None
        self.t_proc = None
        self.t_rec = None
        self.t_roll = None

        # Events gehören jetzt zur App
        self.stop_event = threading.Event()
        self.record_event = threading.Event()

        # Replay gehört jetzt zur App
        self.replay_buffer = deque(maxlen=int(60 * config.replay_seconds))
        self.replay_lock = threading.Lock()
        self.history_frames = deque()

        self._replay_last_store = 0.0
        self._replay_interval = 1.0 / 60.0
        self._max_replay_frames = int(60 * config.replay_seconds)

        self.api = Flask(__name__)
        self.last_analysis = {}

        self.api.add_url_rule(
            "/test",
            "test",
            self.api_test,
            methods=["GET"]
        )
        self.api.add_url_rule(
            "/goal",
            "goal",
            self.api_goal,
            methods=["GET"]
        )
        self.api.add_url_rule(
            "/goal/video/<path:filename>",
            "goal_video",
            self.api_goal_video,
            methods=["GET"]
        )
        self.detector = BallDetector(debug=False)

        self.analyzer = KickerAnalyzer(
            goals_cfg=vision.goals,
            field_cfg=vision.field,
        )


    def api_test(self):
        print("TEST")
        return jsonify({
            "status": "ok"
        })

    def _invalidate_rolling_video(self):
        with self.rolling_video_lock:
            self.rolling_video_path = None
            self.rolling_video_ready_time = 0.0

    def _request_shutdown(self, signum=None, _frame=None):
        signal_name = signal.Signals(signum).name if signum else "shutdown"
        print(f"\n🛑 Stop angefordert ({signal_name})")
        self.stop_event.set()

    def api_loop(self):
        self.api.run(
            host="0.0.0.0",
            port=5000,
            threaded=True,
            use_reloader=False
        )

    def api_goal(self):
        # Es gibt nur noch ein Goal-Video: den letzten 3-Sekunden-Puffer.
        with self.goal_video_lock:
            video_path = self.goal_video_path
            preparing = (
                self.goal_video_thread is not None
                and self.goal_video_thread.is_alive()
            )

        # Ein altes Ringsegment darf nach einem Kamerastillstand nicht erneut
        # als aktuelles Torvideo ausgeliefert werden. Bei laufender Kamera
        # wird spätestens alle drei Sekunden ein neues Segment veröffentlicht.
        if not preparing:
            self._start_goal_video_preparation(force=True)
            preparing = True

        if preparing:
            self.goal_video_ready.wait(timeout=6.0)
            with self.goal_video_lock:
                video_path = self.goal_video_path

        speed = float(self.goal_video_speed)
        frame_count = GOAL_VIDEO_FRAME_LIMIT

        if video_path is None:
            return jsonify({
                "status": "error",
                "message": "Die letzten 3 Sekunden konnten nicht als Video erstellt werden",
            }), 503

        video_filename = Path(video_path).name
        # Relative URL zur Vermeidung von falschem Host, 127.0.0.1 und
        # Mixed-Content-Problemen bei einem HTTPS-Frontend.
        video_url = url_for("goal_video", filename=video_filename)

        return jsonify({
            "ball_speed_kmh": speed,
            "video_url": video_url,
            "video_filename": video_filename,
            "video_frame_count": frame_count,
            "video_fps": self.goal_video_fps,
        })

    def _queue_rolling_video_frame(self, image, now):
        if now - self.rolling_last_submit < 1.0 / GOAL_VIDEO_FPS:
            return

        image = image.copy()
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_BayerRG2BGR)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        width = min(image.shape[1], GOAL_VIDEO_MAX_WIDTH)
        height = int(round(image.shape[0] * width / image.shape[1]))
        height -= height % 2
        if image.shape[1] != width or image.shape[0] != height:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

        try:
            if self.rolling_video_queue.full():
                self.rolling_video_queue.get_nowait()
            self.rolling_video_queue.put_nowait(image)
            self.rolling_last_submit = now
        except Empty:
            pass

    def _rolling_video_encoder_loop(self):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            print("⚠️ FFmpeg nicht gefunden; Ringpuffer ist deaktiviert")
            return

        process = None
        try:
            first_image = self.rolling_video_queue.get(timeout=10.0)
            height, width = first_image.shape[:2]
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pattern = Analyse_Video_Path / f"goal_ring_{stamp}_%03d.webm"
            playlist = Analyse_Video_Path / f"goal_ring_{stamp}.csv"

            command = [
                ffmpeg, "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-video_size", f"{width}x{height}",
                "-framerate", str(GOAL_VIDEO_FPS), "-i", "-",
                "-an", "-c:v", "libvpx", "-deadline", "realtime",
                "-cpu-used", "8", "-threads", "2",
                "-g", str(GOAL_VIDEO_FPS),
                "-crf", "20", "-b:v", "0", "-pix_fmt", "yuv420p",
                "-f", "segment", "-segment_time", str(GOAL_VIDEO_DURATION_SECONDS),
                "-segment_wrap", "3", "-reset_timestamps", "1",
                "-segment_list", str(playlist), "-segment_list_type", "csv",
                "-segment_list_size", "3", "-segment_format", "webm",
                str(pattern),
            ]

            # FFmpeg darf die Kamera-/Analyse-Threads auf dem Orange Pi
            # nicht verdrängen. Nutze, falls vorhanden, die grossen
            # CPU-Kerne 4-7; taskset ist auf dem Pi optional.
            taskset = shutil.which("taskset")
            if taskset:
                command = [taskset, "-c", "4-7"] + command

            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            print(
                f"[RING] FFmpeg gestartet: {width}x{height} "
                f"@ {GOAL_VIDEO_FPS} FPS"
            )

            pending = first_image
            published_name = None
            while not self.stop_event.is_set():
                if process.poll() is not None:
                    error_output = process.stderr.read().decode(errors="replace")
                    raise RuntimeError(
                        f"FFmpeg beendet (code={process.returncode}): {error_output}"
                    )

                process.stdin.write(np.ascontiguousarray(pending).tobytes())
                try:
                    pending = self.rolling_video_queue.get(timeout=0.5)
                except Empty:
                    continue

                if playlist.exists():
                    rows = [line for line in playlist.read_text().splitlines() if line.strip()]
                    if rows:
                        latest_name = rows[-1].split(",", 1)[0]
                        latest_path = Path(latest_name)
                        if not latest_path.is_absolute():
                            latest_path = Analyse_Video_Path / latest_path
                        if (
                            latest_path.exists()
                            and latest_path.stat().st_size > 0
                            and latest_path.name != published_name
                        ):
                            with self.rolling_video_lock:
                                self.rolling_video_path = str(latest_path)
                                self.rolling_video_list = playlist
                                self.rolling_video_ready_time = time.monotonic()
                            print(f"[RING] Segment bereit: {latest_path.name}")
                            published_name = latest_path.name
        except Exception as exc:
            print(f"⚠️ Rolling-Video-Encoder beendet: {exc}")
        finally:
            if process is not None:
                try:
                    process.stdin.close()
                except Exception:
                    pass
                try:
                    process.terminate()
                    process.wait(timeout=2)
                except Exception:
                    try:
                        process.kill()
                    except Exception:
                        pass

    def _start_goal_video_preparation(self, force=False):
        with self.goal_video_lock:
            if (
                self.goal_video_thread is not None
                and self.goal_video_thread.is_alive()
            ):
                return

            # Beim Polling kein neues Video starten, solange bereits ein
            # fertiges Video vorhanden ist.
            if self.goal_video_path is not None and not force:
                return

            with self.history_lock:
                source_frames = list(self.history_frames)

            # Den kompletten 3-Sekunden-Puffer verwenden. Falls die Quelle
            # mehr als 60 FPS liefert, werden Frames gleichmaessig uebersprungen.
            if len(source_frames) > GOAL_VIDEO_FRAME_LIMIT:
                selected_indices = np.linspace(
                    0,
                    len(source_frames) - 1,
                    GOAL_VIDEO_FRAME_LIMIT,
                    dtype=int,
                )
                source_frames = [source_frames[i] for i in selected_indices]

            frames = []
            for source_frame in source_frames:
                frame = copy.copy(source_frame)
                frame.frame = source_frame.frame.copy()
                frames.append(frame)

            self.goal_video_path = None
            self.goal_video_speed = float(self.analyzer.speed_display)
            self.goal_video_ready.clear()

            self.goal_video_thread = threading.Thread(
                target=self._prepare_goal_video,
                args=(frames, self.goal_video_speed),
                daemon=True,
            )
            self.goal_video_thread.start()

    def _prepare_goal_video(self, frames, speed):
        try:
            # Erst mit dem vollstaendigen 3-Sekunden-Puffer kodieren. Wenn
            # der Tor-Trigger sehr frueh kommt, weiter auf neue Kameraframes
            # warten, statt ein zu kurzes Video zu veroeffentlichen.
            deadline = time.perf_counter() + 5.0
            while len(frames) < GOAL_VIDEO_FRAME_LIMIT and time.perf_counter() < deadline:
                time.sleep(0.02)
                with self.history_lock:
                    current_frames = list(self.history_frames)
                if len(current_frames) >= GOAL_VIDEO_FRAME_LIMIT:
                    frames = current_frames
                    break

            if len(frames) < GOAL_VIDEO_FRAME_LIMIT:
                raise RuntimeError(
                    f"Zu wenige Frames fuer Goal-Video: {len(frames)}/"
                    f"{GOAL_VIDEO_FRAME_LIMIT}"
                )

            if len(frames) > GOAL_VIDEO_FRAME_LIMIT:
                selected_indices = np.linspace(
                    0,
                    len(frames) - 1,
                    GOAL_VIDEO_FRAME_LIMIT,
                    dtype=int,
                )
                frames = [frames[i] for i in selected_indices]

            copied_frames = []
            for source_frame in frames:
                frame = copy.copy(source_frame)
                frame.frame = source_frame.frame.copy()
                copied_frames.append(frame)
            frames = copied_frames

            video_path = self.create_goal_video(frames)
            with self.goal_video_lock:
                self.goal_video_path = video_path
                self.goal_video_speed = speed
                self.goal_video_frame_count = len(frames)
        except Exception as exc:
            print(f"Goal-Video Vorbereitung fehlgeschlagen: {exc}")
        finally:
            self.goal_video_ready.set()

    def api_goal_video(self, filename):
        return send_from_directory(
            str(Analyse_Video_Path),
            Path(filename).name,
            mimetype="video/webm",
            as_attachment=False,
        )

    def create_goal_video(self, frames):

        if not frames:
            raise ValueError("Keine Frames vorhanden")
        if len(frames) != GOAL_VIDEO_FRAME_LIMIT:
            raise ValueError(
                f"Goal-Video braucht {GOAL_VIDEO_FRAME_LIMIT} Frames, "
                f"erhalten: {len(frames)}"
            )

        output_dir = Analyse_Video_Path
        output_dir.mkdir(parents=True, exist_ok=True)

        # VP8/WebM ist direkt mit Chrome und Edge kompatibel. Ein fester
        # Dateiname sorgt dafuer, dass immer nur der letzte Rueckblick
        # vorhanden ist.
        video_path = output_dir / "goal_latest.webm"

        first_frame = frames[0].frame
        height, width = first_frame.shape[:2]

        # Das Zielvideo hat bewusst eine feste Zeitbasis: 180 Frames bei
        # 60 FPS ergeben exakt 3 Sekunden, unabhaengig von der Kamera-FPS.
        video_fps = float(GOAL_VIDEO_FPS)
        self.goal_video_fps = video_fps

        # Das Analysevideo muss nicht in voller Kameraaufloesung kodiert
        # werden. Eine kleinere Ausgabe beschleunigt die VP8-Kodierung auf
        # dem Raspberry Pi deutlich.
        output_width = min(width, GOAL_VIDEO_MAX_WIDTH)
        output_height = int(round(height * output_width / width))
        output_height -= output_height % 2

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                "FFmpeg nicht gefunden. Bitte installieren: sudo apt install ffmpeg"
            )

        temp_path = video_path.with_suffix(".tmp.webm")
        command = [
            ffmpeg,
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-video_size", f"{output_width}x{output_height}",
            "-framerate", str(GOAL_VIDEO_FPS),
            "-i", "-",
            "-an",
            "-c:v", "libvpx",
            "-deadline", "realtime",
            "-cpu-used", "8",
            "-crf", "10",
            "-b:v", "0",
            "-pix_fmt", "yuv420p",
            "-f", "webm",
            str(temp_path),
        ]

        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        try:
            for frame in frames:
                image = frame.frame

                # FFmpeg bekommt immer ein 3-kanaliges BGR-Bild.
                if image.ndim == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_BayerRG2BGR)
                elif image.ndim == 3 and image.shape[2] == 1:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

                # Die History kann Rohbilder enthalten, waehrend das
                # Livebild bereits in processing_loop entzerrt wird. Das
                # Goal-Video muss denselben Distortion-Schritt erhalten.
                # Erst danach verkleinern, damit die Kalibrierung weiterhin
                # auf der Originalaufloesung (1100x650) arbeitet.
                image = self.dist.apply(image)

                if image.shape[1] != output_width or image.shape[0] != output_height:
                    image = cv2.resize(
                        image,
                        (output_width, output_height),
                        interpolation=cv2.INTER_AREA,
                    )
                process.stdin.write(np.ascontiguousarray(image).tobytes())

            process.stdin.close()
            error_output = process.stderr.read().decode(errors="replace")
            return_code = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise

        if return_code != 0:
            raise RuntimeError(f"FFmpeg-Videoerzeugung fehlgeschlagen: {error_output}")

        temp_path.replace(video_path)

        print(f"✅ Goal Video erstellt: {video_path}")

        return str(video_path)

    @staticmethod
    def _get_start_mode() -> bool:
        mode = input("1 = Neue Gesamtkalibrierung | 2 = Laden: ").strip()
        return mode == "1"


    def tune_distortion_live(self):
        if HEADLESS:
            print("❌ HEADLESS → kein Distortion UI")
            return

        ret, frame = self.video.read()

        if not ret or frame is None:
            print("❌ Kein Frame verfügbar")
            return

        ui = DistortionTunerUI(self.dist.camera)

        while True:
            ui.update_camera_if_needed()

            corrected = self.dist.apply(frame.frame)

            cv2.imshow("Distortion Tuning", corrected)

            key = cv2.waitKey(15) & 0xFF

            if key == 27:
                break

            if key == ord("s"):
                ui.save_to_config(config.config_path)

        cv2.destroyWindow("Distortion Tuning")

    def _load_or_calibrate(self) -> None:
        do_calibration = self._get_start_mode()

        if do_calibration:
            self.tune_distortion_live()

            ret, frame = self.video.read()

            if not ret or frame is None:
                raise RuntimeError("❌ Kein Frame nach Distortion-Tuning")

            frame.frame = cv2.cvtColor(frame.frame, cv2.COLOR_BayerRG2BGR)
            frame.frame = self.dist.apply(frame.frame)

            pipeline = CalibrationPipeline()
            self.cfg = pipeline.run(frame.frame)

            vision.save()

            print("✅ Kalibrierung gespeichert:", str(vision.DEFAULT_PATH))

        else:
            cfg = JsonStore.load(str(config.config_path))

            if cfg is None:
                raise RuntimeError(f"❌ Kalibrierungsdatei nicht gefunden: {str(config.config_path)}")

            self.cfg = cfg

            print("📁 Kalibrierung geladen:", str(config.config_path))

        self.video.reset()

    # -------------------------------------------------
    # Camera Loop
    # -------------------------------------------------

    def camera_loop(self):
        dropped = 0

        while not self.stop_event.is_set():
            try:
                ret, objframe = self.video.read()

                if not ret or objframe is None:
                    self._invalidate_rolling_video()
                    continue

                # Frame für die letzten 3 Sekunden speichern
                self.history_frames.append(objframe)

                # Alles älter als 3 Sekunden entfernen
                while self.history_frames and objframe.timestamp - self.history_frames[0].timestamp > 3:
                    self.history_frames.popleft()

                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                        dropped += 1
                    except Empty:
                        pass

                self.frame_queue.put(objframe)

                # Aufnahme bekommt bewusst eine Kopie vom Rohbild
                # damit processing_loop das Objekt nicht vorher verändert
                if self.record_event.is_set():
                    try:
                        self.recording_queue.put(objframe)
                    except Exception as e:
                        print(f"⚠️ Recording Queue Fehler: {e}")

            except Exception as e:
                print(f"❌ Camera-Loop Fehler: {e}")
                traceback.print_exc()
                self.stop_event.set()
                break

    def correction_loop(self):

        while not self.stop_event.is_set():

            try:
                frame = self.raw_queue.get(timeout=0.5)

                frame.frame = cv2.cvtColor(
                    frame.frame,
                    cv2.COLOR_BayerRG2BGR
                )

                frame.frame = self.dist.apply(
                    frame.frame
                )

                with self.history_lock:

                    self.history_frames.append(frame)

                    while (
                            self.history_frames
                            and frame.timestamp - self.history_frames[0].timestamp > 3
                    ):
                        self.history_frames.popleft()

                if self.processed_queue.full():
                    try:
                        self.processed_queue.get_nowait()
                    except Empty:
                        pass

                self.processed_queue.put(frame)

            except Empty:
                continue

            except Exception as e:
                print(f"❌ Correction-Loop Fehler: {e}")
                traceback.print_exc()
                self.stop_event.set()
                break

    # -------------------------------------------------
    # Processing Loop
    # -------------------------------------------------
    def processing_loop(self):
        last_log = time.perf_counter()
        n = 0
        stage_time = {
            "cvt": 0.0,
            "dist": 0.0,
            "detect": 0.0,
            "overlay": 0.0,
            "analyze": 0.0,
        }
        skipped_processing_frames = 0

        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.5)

                # Bei einer 200-FPS-Kamera nicht mit alten Frames hinterher-
                # laufen. Fuer die Live-Analyse ist immer der aktuellste
                # verfuegbare Frame relevant.
                while True:
                    try:
                        frame = self.frame_queue.get_nowait()
                        skipped_processing_frames += 1
                    except Empty:
                        break

                stage_start = time.perf_counter()
                frame.frame = cv2.cvtColor(frame.frame, cv2.COLOR_BayerRG2BGR)
                stage_time["cvt"] += time.perf_counter() - stage_start

                stage_start = time.perf_counter()
                frame.frame = self.dist.apply(frame.frame)
                stage_time["dist"] += time.perf_counter() - stage_start

                now = time.perf_counter()
                n += 1

                if now - self._replay_last_store >= self._replay_interval:
                    with self.replay_lock:
                        self.replay_buffer.append((frame.frame.copy(), frame.timestamp))

                        if len(self.replay_buffer) > self._max_replay_frames:
                            self.replay_buffer.popleft()

                    self._replay_last_store = now

                stage_start = time.perf_counter()
                meas = self.detector.detect(frame)
                stage_time["detect"] += time.perf_counter() - stage_start

                stage_start = time.perf_counter()
                self.analyzer.draw_overlay(frame)
                stage_time["overlay"] += time.perf_counter() - stage_start

                if self.display_queue.full():
                    try:
                        self.display_queue.get_nowait()
                    except Empty:
                        pass

                self.display_queue.put(frame.frame)
                goal = False

                # Auch Frames ohne Ball muessen an den Analyzer gehen: Nur
                # so kann er ein Tor bestaetigen, wenn der Ball verschwindet.
                stage_start = time.perf_counter()
                goal = self.analyzer.process_frame(frame)
                stage_time["analyze"] += time.perf_counter() - stage_start

                if goal and any(goal):
                    self.play_last_seconds()

                if now - last_log >= 1.0:
                    dt = now - last_log
                    host_fps = n / dt

                    print(
                        f"[CAM] AnalysierenFPS={host_fps:.1f} | "
                        f"cvt={stage_time['cvt'] / n * 1000:.1f}ms "
                        f"dist={stage_time['dist'] / n * 1000:.1f}ms "
                        f"detect={stage_time['detect'] / n * 1000:.1f}ms "
                        f"overlay={stage_time['overlay'] / n * 1000:.1f}ms "
                        f"analyze={stage_time['analyze'] / n * 1000:.1f}ms "
                        f"uebersprungen={skipped_processing_frames}"
                    )

                    last_log = now
                    n = 0
                    skipped_processing_frames = 0
                    for key in stage_time:
                        stage_time[key] = 0.0

            except Empty:
                continue

            except Exception as e:
                print(f"❌ Processing-Loop Fehler: {e}")
                self.stop_event.set()
                break

    # -------------------------------------------------
    # Recording Loop
    # -------------------------------------------------
    def recording_loop(self):
        raw_frames = []
        raw_times = []

        recording = False
        t_start = None

        try:
            while not self.stop_event.is_set():
                try:
                    want_record = self.record_event.is_set()

                    if want_record and not recording:
                        print("▶️ RAW Recording gestartet")

                        raw_frames.clear()
                        raw_times.clear()

                        t_start = time.perf_counter()
                        recording = True

                    if not want_record and recording:
                        print(f"💾 SAVE START {len(raw_frames)}")

                        self._start_save_thread(raw_frames, raw_times)

                        raw_frames.clear()
                        raw_times.clear()

                        recording = False
                        t_start = None

                    if not recording:
                        time.sleep(0.01)
                        continue

                    raw_frame = self.recording_queue.get(timeout=0.2)

                    raw_frames.append(raw_frame.copy())
                    raw_times.append(time.perf_counter() - t_start)

                except Empty:
                    continue

                except Exception as e:
                    print(f"❌ Recording-Loop Fehler: {e}")
                    self.stop_event.set()
                    break

        finally:
            # Falls während laufender Aufnahme beendet wird,
            # trotzdem alles speichern, was bereits aufgenommen wurde
            if recording and len(raw_frames) > 0:
                print(f"💾 SAVE BEIM BEENDEN {len(raw_frames)}")
                self._start_save_thread(raw_frames, raw_times)

    def _start_save_thread(self, raw_frames, raw_times):
        frames_copy = raw_frames.copy()
        times_copy = raw_times.copy()

        threading.Thread(
            target=self._save_raw_async,
            args=(frames_copy, times_copy),
            daemon=True
        ).start()

    def _save_raw_async(self, frames, times):
        if not frames:
            print("⚠️ Keine Frames zum Speichern vorhanden")
            return

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = Analyse_Video_Path / f"session_raw_{ts}.npz"

        try:
            print("💾 SAVE THREAD START")

            np.savez(
                filename,
                frames=np.stack(frames),
                times=np.array(times)
            )

            print(f"✅ SAVE DONE: {filename}")

        except Exception as e:
            print(f"❌ Fehler beim Speichern der Aufnahme: {e}")

    # -------------------------------------------------
    # Replay
    # -------------------------------------------------
    def play_last_seconds(self):
        with self.replay_lock:
            if len(self.replay_buffer) < 2:
                print("⚠️ Nicht genug Frames für Replay")
                return

            frames = list(self.replay_buffer)

        if HEADLESS:
            print("⚠️ HEADLESS aktiv → lokales Replay wird nicht angezeigt")
            return

        gui_named_window("Replay (letzte Sekunden)")

        target_fps = 120
        target_dt = 1.0 / target_fps

        t_start = frames[0][1]
        t_end = frames[-1][1]

        play_start = time.perf_counter()
        idx = 0

        while not self.stop_event.is_set():
            now = time.perf_counter()
            play_time = t_start + (now - play_start)

            if play_time >= t_end:
                break

            while idx + 1 < len(frames) and frames[idx + 1][1] <= play_time:
                idx += 1

            frame, _ = frames[idx]

            gui_imshow("Replay (letzte Sekunden)", frame)

            if gui_wait_key(int(target_dt * 1000)) == 27:
                break

        gui_destroy_window("Replay (letzte Sekunden)")

    # -------------------------------------------------
    # Run
    # -------------------------------------------------
    def run(self) -> bool:
        restart_requested = False

        try:
            # Signale kommen immer im Main-Thread an. Dadurch funktioniert
            # Ctrl+C auch im Headless-Betrieb und beendet alle Worker sauber.
            signal.signal(signal.SIGINT, self._request_shutdown)
            if hasattr(signal, "SIGTERM"):
                signal.signal(signal.SIGTERM, self._request_shutdown)

            self.stop_event.clear()
            self.record_event.clear()
            self._load_or_calibrate()

            assert self.cfg is not None

            self.t_api = threading.Thread(
                target=self.api_loop,
                daemon=True
            )
            self.t_cam = threading.Thread(
                target=self.camera_loop,
                daemon=True
            )

            self.t_proc = threading.Thread(
                target=self.processing_loop,
                daemon=True
            )

            self.t_rec = threading.Thread(
                target=self.recording_loop,
                daemon=True
            )

            self.t_cam.start()
            self.t_proc.start()
            self.t_rec.start()
            self.t_api.start()

            print("⌨️  Taste 'r' = Aufnahme Start/Stop | ESC/q = Ende | P = Neustart")

            if HEADLESS:
                print("🟢 HEADLESS=1 → keine OpenCV-GUI")

                while not self.stop_event.is_set():
                    time.sleep(0.2)

            else:
                while not self.stop_event.is_set():
                    try:
                        frame = self.display_queue.get(timeout=0.5)
                    except Empty:
                        continue

                    gui_imshow("Kicker Analyse", frame)

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        print("KEY → Ende")
                        break

                    elif key == ord("r"):
                        if self.record_event.is_set():
                            self.record_event.clear()
                            print("KEY → Recording OFF")
                        else:
                            self.record_event.set()
                            print("KEY → Recording ON")

                    elif key == 27:
                        print("KEY → ESC Ende")
                        break

                    elif key == ord("P"):
                        print("KEY → Neustart")
                        restart_requested = True
                        break

        except KeyboardInterrupt:
            print("🛑 Programm durch Benutzer beendet")

        except Exception as e:
            print(f"❌ Unerwarteter Fehler in app.run(): {e}")

        finally:
            self.stop_event.set()
            self.record_event.clear()

            # Kamera vor dem Thread-Join freigeben, damit ein blockiertes
            # RetrieveResult bzw. eine Grab-Recovery sofort abgebrochen wird.
            try:
                self.video.release()
            except BaseException as e:
                print(f"[SHUTDOWN] Videoquelle konnte nicht sofort freigegeben werden: {e}")

            self._join_threads()

            try:
                self.video.release()
                print("✅ Kamera/Videoquelle freigegeben")
            except Exception as e:
                print(f"⚠️ Videoquelle konnte nicht freigegeben werden: {e}")

            try:
                gui_destroy_all()
            except Exception:
                pass

        return restart_requested

    def _join_threads(self):
        for thread in [self.t_cam, self.t_proc, self.t_rec, self.t_roll, self.t_api]:
            if thread is None:
                continue

            try:
                thread.join(timeout=0.3)
            except Exception:
                pass


# -------------------------------------------------
# Main
# -------------------------------------------------
def _create_camera_with_retry(camera_config):
    last_error = None
    for attempt in range(1, 16):
        try:
            return BaslerSource(camera_config)
        except Exception as exc:
            last_error = exc
            print(
                f"[CAM-START] Kamera nicht verfuegbar "
                f"(Versuch {attempt}/15): {exc}"
            )
            if attempt < 15:
                time.sleep(2.0)
    raise RuntimeError(
        f"Basler-Kamera konnte nach 15 Versuchen nicht geoeffnet werden: {last_error}"
    )


if __name__ == "__main__":
    if not _acquire_app_lock():
        sys.exit(1)

    while True:
        source = None
        restart_requested = False

        try:
            if config.use_camera:
                try:
                    source = _create_camera_with_retry(config.camera)
                    print("📷 Quelle: Basler Kamera")

                except Exception as e:
                    print(f"❌ Kamera nicht verfügbar: {e}")
                    print("➡️ Fallback auf RawReplaySource")
                    raise RuntimeError(
                        "Basler-Kamera konnte nicht geoeffnet werden; "
                        "Replay-Fallback ist nicht verfuegbar."
                    ) from e

            else:
                if not VIDEO_PATH.exists():
                    raise FileNotFoundError(f"Replay-Datei fehlt: {VIDEO_PATH}")
                source = RawReplaySource(str(VIDEO_PATH))

            app = KickerApp(source)

            restart_requested = app.run()

        except KeyboardInterrupt:
            print("🛑 Programm durch Benutzer beendet")
            break

        except Exception as e:
            print(f"❌ Unerwarteter Fehler im Main-Block: {e}")

        finally:
            if source is not None:
                try:
                    source.release()
                    print("✅ Kamera freigegeben")
                except Exception as e:
                    print(f"⚠️ Fehler beim Freigeben der Kamera: {e}")

            source = None

            try:
                gui_destroy_all()
            except Exception:
                pass

            time.sleep(1)

        if not restart_requested:
            break
