
import cv2
import copy
import os
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

    def api_loop(self):
        self.api.run(
            host="0.0.0.0",
            port=5000,
            threaded=True,
            use_reloader=False
        )

    def api_goal(self):
        # Das Video wird im Hintergrund vorbereitet. Die HTTP-Anfrage muss
        # nicht mehr auf Analyse oder Videokodierung warten.
        self._start_goal_video_preparation()

        with self.goal_video_lock:
            video_path = self.goal_video_path
            speed = self.goal_video_speed
            frame_count = self.goal_video_frame_count

        if video_path is None:
            return jsonify({
                "status": "preparing",
                "message": "Goal-Video wird noch vorbereitet",
                "retry_after_ms": 250,
            }), 202

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

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # VP8/WebM ist direkt mit Chrome und Edge kompatibel.
        video_path = output_dir / f"goal_{timestamp}.webm"

        first_frame = frames[0].frame
        height, width = first_frame.shape[:2]

        # Die Kamera-FPS aus den echten Zeitstempeln bestimmen. So wird ein
        # 30-FPS- oder ungleichmaessiger Stream nicht kuenstlich mit 60 FPS
        # abgespielt.
        timestamps = np.asarray(
            [float(frame.timestamp) for frame in frames],
            dtype=np.float64,
        )
        elapsed = timestamps[-1] - timestamps[0]
        if elapsed > 0 and len(frames) > 1:
            video_fps = (len(frames) - 1) / elapsed
            video_fps = max(1.0, min(float(GOAL_VIDEO_FPS), video_fps))
        else:
            video_fps = float(GOAL_VIDEO_FPS)
        self.goal_video_fps = video_fps

        # Das Analysevideo muss nicht in voller Kameraaufloesung kodiert
        # werden. Eine kleinere Ausgabe beschleunigt die VP8-Kodierung auf
        # dem Raspberry Pi deutlich.
        output_width = min(width, GOAL_VIDEO_MAX_WIDTH)
        output_height = int(round(height * output_width / width))

        fourcc = cv2.VideoWriter_fourcc(*"VP80")

        writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            video_fps,
            (output_width, output_height)
        )

        if not writer.isOpened():
            raise RuntimeError("WebM/VP8-Video konnte nicht erstellt werden")

        try:
            for frame in frames:
                image = frame.frame

                # Bei der Basler-Kamera kann der Snapshot noch als ein-
                # kanaliges BayerRG8-Bild vorliegen. VideoWriter benoetigt
                # fuer die Ausgabe ein 3-kanaliges BGR-Bild.
                if image.ndim == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_BayerRG2BGR)
                elif image.ndim == 3 and image.shape[2] == 1:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

                if image.shape[1] != output_width:
                    image = cv2.resize(
                        image,
                        (output_width, output_height),
                        interpolation=cv2.INTER_AREA,
                    )
                writer.write(image)

        finally:
            writer.release()

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

        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.5)

                frame.frame = cv2.cvtColor(frame.frame, cv2.COLOR_BayerRG2BGR)
                frame.frame = self.dist.apply(frame.frame)

                now = time.perf_counter()
                n += 1

                if now - self._replay_last_store >= self._replay_interval:
                    with self.replay_lock:
                        self.replay_buffer.append((frame.frame.copy(), frame.timestamp))

                        if len(self.replay_buffer) > self._max_replay_frames:
                            self.replay_buffer.popleft()

                    self._replay_last_store = now

                meas = self.detector.detect(frame)

                self.analyzer.draw_overlay(frame)

                if self.display_queue.full():
                    try:
                        self.display_queue.get_nowait()
                    except Empty:
                        pass

                self.display_queue.put(frame.frame)

                goal = False

                if meas is True:
                    goal = self.analyzer.process_frame(frame)

                if goal is True:
                    self._start_goal_video_preparation(force=True)
                    self.play_last_seconds()

                if now - last_log >= 1.0:
                    dt = now - last_log
                    host_fps = n / dt

                    print(f"[CAM] AnalysierenFPS={host_fps:.1f} | ")

                    last_log = now
                    n = 0

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

            self._join_threads()

            try:
                gui_destroy_all()
            except Exception:
                pass

        return restart_requested

    def _join_threads(self):
        for thread in [self.t_cam, self.t_proc, self.t_rec, self.t_api]:
            if thread is None:
                continue

            try:
                thread.join(timeout=1)
            except Exception:
                pass


# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    while True:
        source = None
        restart_requested = False

        try:
            if config.use_camera:
                try:
                    source = BaslerSource(config.camera)
                    print("📷 Quelle: Basler Kamera")

                except Exception as e:
                    print(f"❌ Kamera nicht verfügbar: {e}")
                    print("➡️ Fallback auf RawReplaySource")
                    source = RawReplaySource(str(VIDEO_PATH))

            else:
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
