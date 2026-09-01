
import cv2
import copy
import json
import os
import signal
import subprocess
import shutil
import ssl
import sys
import time
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Dict, Optional
from flask import Flask, jsonify, request, send_from_directory, url_for
from urllib import request as urllib_request

import numpy as np
import traceback

try:
    import requests
except ImportError:  # pragma: no cover - fallback is used on minimal installs
    requests = None

try:
    import fcntl
except ImportError:
    fcntl = None

from .config import AppConfig
from .RawReplaySource import RawReplaySource
from .BaslerSource import BaslerSource
from .Objects.Frames import Frame
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
# Fuer die Geschwindigkeits-/Tor-Nachanalyse werden alle Rohframes aus der
# letzten Sekunde bis einschliesslich des ersten Frames im Tor abgelegt.
GOAL_PRE_NPZ_DURATION_SECONDS = 1.0
# 60 Gesamtframes fuer drei Sekunden entsprechen 20 FPS.
GOAL_VIDEO_FRAME_LIMIT = 60
GOAL_VIDEO_FPS = GOAL_VIDEO_FRAME_LIMIT / GOAL_VIDEO_DURATION_SECONDS
GOAL_VIDEO_MAX_WIDTH = 640

# Die Live-Analyse darf alte Frames ueberspringen, aber nicht dauerhaft
# verlieren. Nach dieser Anzahl an Live-Frames ohne Ball werden die letzten
# uebersprungenen Rohframes einmal chronologisch nachanalysiert.
RETRO_TRIGGER_NO_BALL_FRAMES = 80
RETRO_MAX_REANALYZE_FRAMES = RETRO_TRIGGER_NO_BALL_FRAMES
RETRO_BUFFER_MAX_FRAMES = 160

# digiKicker kann die URL bei einer anderen Installation per Umgebungsvariable
# aendern. Der Standard entspricht der vom Frontend genannten URL.
DIGIKICKER_POST_DEFAULT_URL = "https://localhost/goal"
DIGIKICKER_POST_TIMEOUT_SECONDS = 2.0

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
        self.goal_video_generation = 0
        self.goal_video_result_delivered = False
        self.goal_video_error: Optional[str] = None
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
        self.replay_buffer = deque(maxlen=int(240 * config.replay_seconds))
        self.replay_lock = threading.Lock()
        self.history_frames = deque()

        # Rohframes aus der Live-Queue, die wegen der Analysegeschwindigkeit
        # uebersprungen wurden. Die vorhandenen Bildreferenzen werden direkt
        # weiterverwendet, damit kein zweiter grosser Bildspeicher entsteht.
        self.skipped_frame_lock = threading.Lock()
        self.skipped_frame_buffer = deque(maxlen=RETRO_BUFFER_MAX_FRAMES)
        self._frame_sequence = 0
        self._last_visible_frame_id = None
        self._last_retro_frame_id = None
        self._no_ball_live_frames = 0
        self._retro_analysis_count = 0

        # Ergebnis des letzten Tors fuer API, POST und Goal-Video.
        self.last_analysis_lock = threading.Lock()
        self._last_posted_goal_event_id = 0
        self.digikicker_post_url = os.environ.get(
            "DIGIKICKER_POST_URL",
            DIGIKICKER_POST_DEFAULT_URL,
        ).strip()
        self.digikicker_post_verify_tls = os.environ.get(
            "DIGIKICKER_POST_VERIFY_TLS",
            "0",
        ).lower() in {"1", "true", "yes", "on"}
        try:
            self.digikicker_post_timeout = float(os.environ.get(
                "DIGIKICKER_POST_TIMEOUT",
                str(DIGIKICKER_POST_TIMEOUT_SECONDS),
            ))
        except ValueError:
            self.digikicker_post_timeout = DIGIKICKER_POST_TIMEOUT_SECONDS

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
        camera_fps = getattr(config.camera, "force_fps", 180) or 180
        table_width_cm = 120.0
        if vision.field is not None:
            table_width_cm = float(vision.field.size_cm.w)
        self.detector = BallDetector(
            hsv_low=tuple(config.ball_detection.get("hsv_low", (100, 110, 90))),
            hsv_high=tuple(config.ball_detection.get("hsv_high", (112, 255, 255))),
            area_min=float(config.ball_detection.get("area_min", 60.0)),
            area_max=float(config.ball_detection.get("area_max", 12000.0)),
            debug=False,
            detection_scale=float(config.ball_detection.get("detection_scale", 0.5)),
            expected_fps=float(camera_fps),
            table_width_cm=table_width_cm,
            radius_min_px=float(config.ball_detection.get("radius_min_px", 5.0)),
            radius_max_px=float(config.ball_detection.get("radius_max_px", 25.0)),
            circularity_min=float(config.ball_detection.get("circularity_min", 0.5)),
            solidity_min=float(config.ball_detection.get("solidity_min", 0.85)),
        )

        self.analyzer = KickerAnalyzer(
            goals_cfg=vision.goals,
            field_cfg=vision.field,
        )


    def api_test(self):
        print("TEST")
        return jsonify({
            "status": "ok"
        })

    def _remember_skipped_frame(self, frame) -> None:
        """Keep a frame removed from the live queue for later re-analysis."""
        if frame is None:
            return
        with self.skipped_frame_lock:
            self.skipped_frame_buffer.append(frame)

    def _take_skipped_frames(self, up_to_frame_id=None):
        """Take skipped frames up to the current live frame in time order."""
        with self.skipped_frame_lock:
            if up_to_frame_id is None:
                selected = list(self.skipped_frame_buffer)
                self.skipped_frame_buffer.clear()
                return selected

            selected = []
            remaining = deque(maxlen=RETRO_BUFFER_MAX_FRAMES)
            for skipped in self.skipped_frame_buffer:
                skipped_id = getattr(skipped, "frame_id", None)
                if skipped_id is None or skipped_id <= up_to_frame_id:
                    selected.append(skipped)
                else:
                    remaining.append(skipped)
            self.skipped_frame_buffer = remaining

        # Eine Rueckanalyse darf die Live-Analyse nicht fuer mehrere Sekunden
        # blockieren. Die aeltesten Frames sind fuer den aktuellen
        # Torverdacht weniger relevant als die letzten Frames vor dem Trigger.
        if len(selected) > RETRO_MAX_REANALYZE_FRAMES:
            selected = selected[-RETRO_MAX_REANALYZE_FRAMES:]

        selected.sort(
            key=lambda item: (
                getattr(item, "frame_id", 0),
                getattr(item, "timestamp", 0.0),
            )
        )
        return selected

    def _has_skipped_frame_before(self, frame_id) -> bool:
        with self.skipped_frame_lock:
            if frame_id is None:
                return bool(self.skipped_frame_buffer)
            return any(
                getattr(skipped, "frame_id", frame_id) <= frame_id
                for skipped in self.skipped_frame_buffer
            )

    def _prepare_and_detect(self, source_frame):
        """Copy, correct and detect one raw frame from the replay buffer."""
        frame = copy.copy(source_frame)
        frame.frame = source_frame.frame.copy()
        if frame.frame.ndim == 2:
            frame.frame = cv2.cvtColor(frame.frame, cv2.COLOR_BayerRG2BGR)
        elif frame.frame.ndim == 3 and frame.frame.shape[2] == 1:
            frame.frame = cv2.cvtColor(frame.frame, cv2.COLOR_GRAY2BGR)
        frame.frame = self.dist.apply(frame.frame)
        self.detector.detect(frame)
        return frame

    def _reanalyze_skipped_frames(self, current_frame_id):
        """Run bounded detection on skipped frames before the live frame."""
        skipped_frames = self._take_skipped_frames(current_frame_id)
        if not skipped_frames:
            return []

        recovered = []
        for skipped in skipped_frames:
            try:
                recovered.append(self._prepare_and_detect(skipped))
            except Exception as exc:
                print(f"[RETRO] Frame konnte nicht analysiert werden: {exc}")

        self._retro_analysis_count += len(recovered)
        return recovered

    def _process_recovered_frames(self, recovered_frames, current_frame):
        """Use recovered positions for speed and explicitly confirm a goal.

        Some recovered frames are older than a live frame already handled by
        the analyzer. They must not be sent through the stateful goal machine
        a second time. Their ball observations are added in timestamp order;
        a detected entry into a goal is then confirmed against the current
        missing/outside frame.
        """
        goal_candidate = None
        for recovered_frame in recovered_frames:
            self.analyzer.observe_recovered_frame(recovered_frame)
            ball = recovered_frame.ball
            if ball is not None:
                recovered_id = getattr(recovered_frame, "frame_id", None)
                if (
                    recovered_id is not None
                    and (
                        self._last_visible_frame_id is None
                        or recovered_id > self._last_visible_frame_id
                    )
                ):
                    self._last_visible_frame_id = recovered_id
            if ball is None or goal_candidate is not None:
                continue

            x, y, radius = ball
            frame_shape = recovered_frame.frame.shape
            if self.analyzer._ball_in_goal_roi(
                x, y, radius, "left", frame_shape
            ):
                goal_candidate = (
                    "left",
                    float(recovered_frame.timestamp),
                )
            elif self.analyzer._ball_in_goal_roi(
                x, y, radius, "right", frame_shape
            ):
                goal_candidate = (
                    "right",
                    float(recovered_frame.timestamp),
                )

        if goal_candidate is None:
            return False, False

        side, landing_timestamp = goal_candidate
        current_ball = getattr(current_frame, "ball", None)
        current_is_still_in_goal = False
        if current_ball is not None:
            x, y, radius = current_ball
            current_is_still_in_goal = self.analyzer._ball_in_goal_roi(
                x,
                y,
                radius,
                side,
                current_frame.frame.shape,
            )

        # The current trigger is either a missing ball or a ball which has
        # returned outside the goal. In both cases the recovered entry is now
        # a complete goal event. If it is still inside, wait for the ordinary
        # goal state machine to see it leave.
        if current_ball is None or not current_is_still_in_goal:
            return self.analyzer.confirm_recovered_goal(side, landing_timestamp)
        return False, False

    def _post_goal_result(self, payload: Dict[str, Any]) -> None:
        """POST a completed goal result to digiKicker without blocking vision."""
        if not self.digikicker_post_url:
            print("[DIGIKICKER] POST deaktiviert (keine URL konfiguriert)")
            return

        headers = {
            "Content-Type": "application/json",
            # Das Frontend-Log nennt 127.0.0.1 als Origin.
            "Origin": "127.0.0.1",
        }
        try:
            if requests is not None:
                response = requests.post(
                    self.digikicker_post_url,
                    json=payload,
                    headers=headers,
                    timeout=self.digikicker_post_timeout,
                    verify=self.digikicker_post_verify_tls,
                )
                response.raise_for_status()
                print(
                    f"[DIGIKICKER] Tor-POST erfolgreich: "
                    f"{response.status_code} {self.digikicker_post_url}"
                )
                return

            # Fallback fuer Installationen ohne requests.
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            post_request = urllib_request.Request(
                self.digikicker_post_url,
                data=body,
                headers=headers,
                method="POST",
            )
            ssl_context = None
            if not self.digikicker_post_verify_tls:
                ssl_context = ssl._create_unverified_context()
            with urllib_request.urlopen(
                post_request,
                timeout=self.digikicker_post_timeout,
                context=ssl_context,
            ) as response:
                if response.status >= 400:
                    raise RuntimeError(f"HTTP {response.status}")
                print(
                    f"[DIGIKICKER] Tor-POST erfolgreich: "
                    f"{response.status} {self.digikicker_post_url}"
                )
        except Exception as exc:
            # Ein nicht erreichbares Frontend darf die Kameraanalyse nicht
            # stoppen. Ein spaeteres Tor erzeugt automatisch den naechsten POST.
            print(f"[DIGIKICKER] Tor-POST fehlgeschlagen: {exc}")

    def _handle_goal(self, goal) -> None:
        if not goal or not any(goal):
            return

        event = dict(self.analyzer.last_goal_event or {})
        side = "left" if goal[0] else "right"
        speed = float(event.get("speed_kmh", self.analyzer.speed_kmh_s))
        event.update({
            "event": "goal",
            "side": event.get("side", side),
            "goal_side": event.get("goal_side", side),
            "goal": event.get("goal", side),
            "position": event.get("position", side),
            "goal_position": event.get("goal_position", side),
            "speed_kmh": speed,
            "ball_speed_kmh": speed,
            "speed": speed,
        })

        event_id = event.get("event_id")
        if event_id is not None and event_id == self._last_posted_goal_event_id:
            return
        if event_id is not None:
            self._last_posted_goal_event_id = event_id

        with self.last_analysis_lock:
            self.last_analysis = event

        # Fuer jedes Tor sofort den vollstaendigen Rohframe-Zeitraum vor dem
        # Einschlag sichern. Das geschieht vor dem asynchronen POST, damit
        # die Torzeit und damit die Auswahl der Frames unveraendert bleiben.
        self._start_goal_npz_save(event)

        # Das POST wird bewusst in einem Daemon-Thread versendet, damit ein
        # langsames oder nicht erreichbares HTTPS-Frontend keine Frames kostet.
        threading.Thread(
            target=self._post_goal_result,
            args=(event.copy(),),
            name="digikicker-goal-post",
            daemon=True,
        ).start()

        print(
            f"[GOAL] {event['side']} | "
            f"Geschwindigkeit={speed:.2f} km/h | "
            f"Score {self.analyzer.score_left}:{self.analyzer.score_right}"
        )

        if not HEADLESS:
            self.play_last_seconds()

    def _start_goal_npz_save(self, event: Dict[str, Any]) -> None:
        """Start saving every raw frame from the second before a goal.

        ``replay_buffer`` is filled in ``camera_loop`` before frames enter
        the processing queue, so it also contains frames which were skipped
        by the live analysis. The snapshot is bounded by the goal timestamp,
        not by the later frame on which the goal state is finally confirmed.
        """
        try:
            goal_timestamp = float(event["goal_timestamp"])
        except (KeyError, TypeError, ValueError):
            print("[GOAL-NPZ] Keine gueltige Torzeit; Speicherung uebersprungen")
            return

        start_timestamp = goal_timestamp - GOAL_PRE_NPZ_DURATION_SECONDS
        with self.replay_lock:
            selected = [
                (image, float(timestamp))
                for image, timestamp in self.replay_buffer
                if start_timestamp <= float(timestamp) <= goal_timestamp
            ]

        if not selected:
            print(
                "[GOAL-NPZ] Keine Rohframes im Zeitfenster "
                f"{start_timestamp:.6f}..{goal_timestamp:.6f}"
            )
            return

        frames = [image for image, _ in selected]
        timestamps = [timestamp for _, timestamp in selected]
        npz_event = dict(event)

        threading.Thread(
            target=self._save_goal_npz_async,
            args=(frames, timestamps, goal_timestamp, npz_event),
            name="goal-npz-save",
            daemon=True,
        ).start()

        print(
            f"[GOAL-NPZ] Speicherung gestartet: {len(frames)} Rohframes "
            f"({GOAL_PRE_NPZ_DURATION_SECONDS:.1f}s vor Tor)"
        )

    def _save_goal_npz_async(
        self,
        frames,
        timestamps,
        goal_timestamp: float,
        event: Dict[str, Any],
    ) -> None:
        """Write one complete pre-goal raw-frame window as an NPZ file."""
        if not frames or not timestamps:
            return

        temp_filename = None
        try:
            first_timestamp = float(timestamps[0])
            relative_times = np.asarray(
                [float(timestamp) - first_timestamp for timestamp in timestamps],
                dtype=np.float64,
            )
            goal_time = float(goal_timestamp) - first_timestamp
            filename = Analyse_Video_Path / (
                "goal_pre_"
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}.npz"
            )
            temp_filename = filename.with_name(f".{filename.name}.tmp")

            # Die Kopie wird erst hier erzeugt, damit der Kamera-Thread beim
            # Ausloesen des Speicherns moeglichst wenig blockiert.
            frame_array = np.stack([np.asarray(frame).copy() for frame in frames])
            # Zuerst in eine temporaere Datei schreiben und erst nach einem
            # vollstaendigen Write umbenennen. So kann kein unvollstaendiges
            # NPZ als gueltige Toraufnahme liegen bleiben.
            with temp_filename.open("wb") as npz_file:
                np.savez(
                    npz_file,
                    frames=frame_array,
                    # RawReplaySource kann diese Zeitachse direkt wiedergeben.
                    times=relative_times,
                    # Zusaetzliche absolute Zeitstempel erleichtern die spaetere
                    # Auswertung, ohne das bestehende Replay-Format zu brechen.
                    timestamps=np.asarray(timestamps, dtype=np.float64),
                    goal_time=np.asarray(goal_time, dtype=np.float64),
                    goal_position=np.asarray(event.get("goal_position", "")),
                    speed_kmh=np.asarray(
                        float(event.get("speed_kmh", 0.0)),
                        dtype=np.float64,
                    ),
                )
            temp_filename.replace(filename)
            print(
                f"[GOAL-NPZ] SAVE DONE: {filename} "
                f"({len(frames)} Frames, Torzeit={goal_time:.6f}s)"
            )
        except Exception as exc:
            print(f"[GOAL-NPZ] Speichern fehlgeschlagen: {exc}")
        finally:
            if temp_filename is not None:
                try:
                    temp_filename.unlink()
                except FileNotFoundError:
                    pass
                except OSError as cleanup_error:
                    print(f"[GOAL-NPZ] Temporärdatei nicht entfernbar: {cleanup_error}")

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

    def _api_goal_fast(self):
        """Start/poll the latest three-second goal video without blocking."""
        refresh = request.args.get("refresh", "").lower() in {
            "1", "true", "yes", "on"
        }
        requested_generation = request.args.get("generation", type=int)

        with self.goal_video_lock:
            video_path = self.goal_video_path
            preparing = (
                self.goal_video_thread is not None
                and self.goal_video_thread.is_alive()
            )
            generation = self.goal_video_generation
            result_delivered = self.goal_video_result_delivered
            error = self.goal_video_error

        # A bare request starts a new snapshot only after the previous result
        # was consumed. A generation/status request only polls and never
        # starts a second encoder job.
        should_start = refresh or (
            requested_generation is None
            and not preparing
            and (video_path is None or result_delivered)
        )
        if should_start:
            generation = self._start_goal_video_preparation(force=True)
            preparing = True

        if requested_generation is not None and requested_generation != generation:
            requested_generation = generation

        if preparing or video_path is None:
            with self.last_analysis_lock:
                latest_goal = dict(self.last_analysis)
            response = {
                "status": "preparing",
                "video_ready": False,
                "video_generation": generation,
                "ball_speed_kmh": float(self.goal_video_speed),
                "retry_after_ms": 100,
                "status_url": url_for("goal", generation=generation),
            }
            if latest_goal:
                response["goal_position"] = latest_goal.get("goal_position")
                response["goal_side"] = latest_goal.get("side")
            if error:
                response["error"] = error
            return jsonify(response), 202

        if error:
            return jsonify({
                "status": "error",
                "video_ready": False,
                "video_generation": generation,
                "message": error,
            }), 503

        # With a bare polling client, expose the completed generation once.
        # The next bare request then starts a fresh three-second snapshot.
        with self.goal_video_lock:
            self.goal_video_result_delivered = True

        speed = float(self.goal_video_speed)
        frame_count = self.goal_video_frame_count or GOAL_VIDEO_FRAME_LIMIT
        with self.last_analysis_lock:
            latest_goal = dict(self.last_analysis)
        video_filename = Path(video_path).name
        try:
            # Keep the filename stable but make every encoded version a new
            # URL so browsers cannot replay a cached old WebM.
            video_version = Path(video_path).stat().st_mtime_ns
        except OSError:
            video_version = time.time_ns()

        response = {
            "status": "ready",
            "video_ready": True,
            "video_generation": generation,
            "ball_speed_kmh": speed,
            "video_url": url_for(
                "goal_video",
                filename=video_filename,
                v=video_version,
            ),
            "video_filename": video_filename,
            "video_frame_count": frame_count,
            "video_fps": self.goal_video_fps,
        }
        if latest_goal:
            response["goal_position"] = latest_goal.get("goal_position")
            response["goal_side"] = latest_goal.get("side")
        return jsonify(response), 200

    def api_goal(self):
        return self._api_goal_fast()

    def _api_goal_legacy(self):
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

    def _snapshot_replay_frames(self):
        """Liefert unveraenderte Rohframes aus den letzten drei Sekunden."""
        with self.replay_lock:
            source_frames = list(self.replay_buffer)

        frames = []
        for image, timestamp in source_frames:
            frame = Frame()
            # Das Array wird im Kamera-Loop nie mehr veraendert. Die Referenz
            # macht den API-Aufruf schnell; kopiert wird erst nach der
            # Auswahl der 60 Zielbilder im Hintergrund.
            frame.frame = image
            frame.timestamp = timestamp
            frames.append(frame)
        return frames

    @staticmethod
    def _select_goal_frames(source_frames):
        """Reduziert beliebig viele Rohframes gleichmaessig auf 60 Bilder."""
        if len(source_frames) <= GOAL_VIDEO_FRAME_LIMIT:
            return list(source_frames)

        selected_indices = np.linspace(
            0,
            len(source_frames) - 1,
            GOAL_VIDEO_FRAME_LIMIT,
            dtype=int,
        )
        return [source_frames[i] for i in selected_indices]

    def _prepare_selected_frames(self, source_frames):
        """Wendet Farbe und Entzerrung erst nach der Frame-Auswahl an."""
        prepared = []
        for source_frame in source_frames:
            frame = copy.copy(source_frame)
            image = source_frame.frame.copy()

            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_BayerRG2BGR)
            elif image.ndim == 3 and image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            frame.frame = self.dist.apply(image)
            prepared.append(frame)
        return prepared

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
                return self.goal_video_generation

            # Beim Polling kein neues Video starten, solange bereits ein
            # fertiges Video vorhanden ist.
            if self.goal_video_path is not None and not force:
                return

            self.goal_video_path = None
            with self.last_analysis_lock:
                latest_goal = dict(self.last_analysis)
            self.goal_video_speed = float(
                latest_goal.get("speed_kmh", self.analyzer.speed_kmh_s)
            )
            self.goal_video_error = None
            self.goal_video_generation += 1
            self.goal_video_result_delivered = False
            self.goal_video_ready.clear()

            self.goal_video_thread = threading.Thread(
                target=self._prepare_goal_video,
                args=(self.goal_video_speed, self.goal_video_generation),
                daemon=True,
            )
            self.goal_video_thread.start()
            return self.goal_video_generation

    def _prepare_goal_video(self, speed, generation):
        try:
            # Erst auf einen echten Drei-Sekunden-Zeitraum warten. Die
            # Reduzierung auf 60 Bilder passiert danach, noch bevor Farbe
            # und Distortion-Korrektur berechnet werden.
            deadline = time.perf_counter() + 5.0
            frames = []
            while time.perf_counter() < deadline:
                current_frames = self._snapshot_replay_frames()
                span = (
                    current_frames[-1].timestamp - current_frames[0].timestamp
                    if len(current_frames) >= 2
                    else 0.0
                )
                if (
                    len(current_frames) >= GOAL_VIDEO_FRAME_LIMIT
                    and span >= GOAL_VIDEO_DURATION_SECONDS
                ):
                    frames = self._select_goal_frames(current_frames)
                    break
                time.sleep(0.02)

            if len(frames) < GOAL_VIDEO_FRAME_LIMIT:
                raise RuntimeError(
                    f"Zu wenige Frames fuer Goal-Video: {len(frames)}/"
                    f"{GOAL_VIDEO_FRAME_LIMIT}"
                )

            frames = self._prepare_selected_frames(frames)

            video_path = self.create_goal_video(frames)
            with self.goal_video_lock:
                if generation == self.goal_video_generation:
                    self.goal_video_path = video_path
                    self.goal_video_speed = speed
                    self.goal_video_frame_count = len(frames)
        except Exception as exc:
            with self.goal_video_lock:
                if generation == self.goal_video_generation:
                    self.goal_video_error = str(exc)
            print(f"Goal-Video Vorbereitung fehlgeschlagen: {exc}")
        finally:
            self.goal_video_ready.set()

    def api_goal_video(self, filename):
        response = send_from_directory(
            str(Analyse_Video_Path),
            Path(filename).name,
            mimetype="video/webm",
            as_attachment=False,
        )
        response.headers["Cache-Control"] = (
            "no-store, no-cache, must-revalidate, max-age=0"
        )
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

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

        # Das Zielvideo hat bewusst eine feste Zeitbasis: 60 Frames bei
        # 20 FPS ergeben exakt 3 Sekunden, unabhaengig von der Kamera-FPS.
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
            "-threads", "2",
            "-crf", "10",
            "-b:v", "0",
            "-pix_fmt", "yuv420p",
            "-f", "webm",
            str(temp_path),
        ]

        # Die Goal-Kodierung darf den Kamera-Grab und die Analyse auf dem
        # Orange Pi nicht verdrängen. Wenn vorhanden, laufen die zwei
        # FFmpeg-Threads deshalb auf den grossen Kernen 4-7.
        taskset = shutil.which("taskset")
        if taskset:
            command = [taskset, "-c", "4-7"] + command

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

                # Die Sequenznummer gilt fuer alle Rohframes, auch wenn ein
                # Frame spaeter aus der kleinen Live-Queue entfernt wird.
                self._frame_sequence += 1
                objframe.frame_id = self._frame_sequence

                # Frame für die letzten 3 Sekunden speichern
                with self.replay_lock:
                    self.replay_buffer.append(
                        (objframe.frame.copy(), objframe.timestamp)
                    )

                # Alles älter als 3 Sekunden entfernen
                with self.replay_lock:
                    while (
                        self.replay_buffer
                        and objframe.timestamp - self.replay_buffer[0][1]
                        > GOAL_VIDEO_DURATION_SECONDS
                    ):
                        self.replay_buffer.popleft()

                if self.frame_queue.full():
                    try:
                        skipped = self.frame_queue.get_nowait()
                        self._remember_skipped_frame(skipped)
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
                # Ein einzelner Fehler beim Lesen/Weiterreichen eines Frames
                # darf die komplette Torerkennung nicht beenden. Die
                # BaslerSource versucht den Stream selbst wiederherzustellen;
                # hier wird nur kurz pausiert, damit bei einem transienten
                # Fehler keine enge Fehler-Schleife entsteht.
                time.sleep(0.05)
                continue

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
                time.sleep(0.05)
                continue

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

                # Fuer die Live-Ansicht bleibt nur der aktuellste Frame
                # relevant. Die uebersprungenen Frames werden aber nicht mehr
                # verworfen, sondern kommen in den Rueckwaertspuffer.
                while True:
                    try:
                        skipped = self.frame_queue.get_nowait()
                        self._remember_skipped_frame(skipped)
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

                stage_start = time.perf_counter()
                meas = self.detector.detect(frame)
                stage_time["detect"] += time.perf_counter() - stage_start

                no_ball_before = self._no_ball_live_frames
                current_frame_id = getattr(frame, "frame_id", None)
                raw_frame_gap = None
                if (
                    current_frame_id is not None
                    and self._last_visible_frame_id is not None
                ):
                    raw_frame_gap = current_frame_id - self._last_visible_frame_id

                if frame.ball is None:
                    self._no_ball_live_frames += 1
                else:
                    # Ein sichtbarer Ball beendet die normale Fehlerserie.
                    self._no_ball_live_frames = 0

                recover_due_to_missing_ball = (
                    frame.ball is None
                    and (
                        self._no_ball_live_frames >= RETRO_TRIGGER_NO_BALL_FRAMES
                        or (
                            raw_frame_gap is not None
                            and raw_frame_gap >= RETRO_TRIGGER_NO_BALL_FRAMES
                            and (
                                self._last_retro_frame_id is None
                                or current_frame_id - self._last_retro_frame_id
                                >= RETRO_TRIGGER_NO_BALL_FRAMES
                            )
                        )
                    )
                )
                # Falls der Ball nach einer kurzen Detektionsluecke wieder
                # auftaucht, werden die inzwischen uebersprungenen Frames
                # ebenfalls vor dem aktuellen Frame chronologisch eingeordnet.
                recover_due_to_ball_return = (
                    frame.ball is not None
                    and (
                        no_ball_before >= 3
                        or (
                            raw_frame_gap is not None
                            and raw_frame_gap >= 3
                            and (
                                self._last_retro_frame_id is None
                                or current_frame_id - self._last_retro_frame_id >= 3
                            )
                        )
                    )
                )
                should_recover = (
                    recover_due_to_missing_ball or recover_due_to_ball_return
                ) and self._has_skipped_frame_before(current_frame_id)

                if should_recover:
                    self._last_retro_frame_id = current_frame_id
                    recovered_frames = self._reanalyze_skipped_frames(current_frame_id)
                    stage_start = time.perf_counter()
                    recovered_goal = self._process_recovered_frames(
                        recovered_frames,
                        frame,
                    )
                    stage_time["analyze"] += time.perf_counter() - stage_start
                    self._handle_goal(recovered_goal)

                    # Der aktuelle Frame ist jetzt der naechste chronologische
                    # Frame. Eine fehlende Detektion zaehlt erneut ab 1.
                    if frame.ball is None:
                        self._no_ball_live_frames = 1

                if frame.ball is not None and current_frame_id is not None:
                    self._last_visible_frame_id = current_frame_id

                # Auch ein Frame ohne Ball muss in den Analyzer: Nur so kann
                # ein Tor nach dem Verschwinden des Balles bestaetigt werden.
                stage_start = time.perf_counter()
                goal = self.analyzer.process_frame(frame)
                stage_time["analyze"] += time.perf_counter() - stage_start
                self._handle_goal(goal)

                stage_start = time.perf_counter()
                self.analyzer.draw_overlay(frame)
                stage_time["overlay"] += time.perf_counter() - stage_start

                if self.display_queue.full():
                    try:
                        self.display_queue.get_nowait()
                    except Empty:
                        pass

                self.display_queue.put(frame.frame)

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
                        f"uebersprungen={skipped_processing_frames} "
                        f"retro={self._retro_analysis_count}"
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
                traceback.print_exc()
                # Die NPZ-Speicherung läuft absichtlich unabhängig vom
                # Analyzer. Ein Fehler bei einem einzelnen Frame darf weder
                # die laufende Analyse noch die nächste Torerkennung stoppen.
                time.sleep(0.01)
                continue

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
                    time.sleep(0.05)
                    continue

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

        raw_frames = []
        for image, timestamp in self._select_goal_frames(frames):
            replay_frame = Frame()
            replay_frame.frame = image
            replay_frame.timestamp = timestamp
            raw_frames.append(replay_frame)
        frames = [
            (frame.frame, frame.timestamp)
            for frame in self._prepare_selected_frames(raw_frames)
        ]

        if HEADLESS:
            print("⚠️ HEADLESS aktiv → lokales Replay wird nicht angezeigt")
            return

        gui_named_window("Replay (letzte Sekunden)")

        target_fps = GOAL_VIDEO_FPS
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


def _find_replay_video() -> Path:
    """Find a usable NPZ replay for camera-less operation."""
    candidates = []

    # test.npz bleibt der bevorzugte, explizite Fallback.
    if VIDEO_PATH.exists():
        candidates.append(VIDEO_PATH)

    # Auf dem Pi werden Aufnahmen vom Recording-Thread in v3/videos abgelegt;
    # lokal liegen sie haeufig noch unter tracking_kicker/videos.
    replay_directories = [config.videos_dir, Analyse_Video_Path]
    if config.debug_video:
        configured_path = Path(config.debug_video)
        if not configured_path.is_absolute():
            for directory in replay_directories:
                candidates.append(directory / configured_path)
        else:
            candidates.append(configured_path)

    for directory in replay_directories:
        candidates.extend(directory.glob("session_raw_*.npz"))

    existing = []
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        existing.append(candidate)

    if not existing:
        searched = ", ".join(str(path) for path in replay_directories)
        raise FileNotFoundError(
            "Kein Fallback-Video gefunden. Erwartet test.npz oder "
            f"session_raw_*.npz in: {searched}"
        )

    # Falls test.npz fehlt, die zuletzt gespeicherte Session verwenden.
    if VIDEO_PATH.resolve() in existing:
        return VIDEO_PATH.resolve()
    return max(existing, key=lambda path: path.stat().st_mtime_ns)


def _create_replay_fallback():
    replay_path = _find_replay_video()
    print(f"[REPLAY] Fallback-Quelle: {replay_path}")
    return RawReplaySource(str(replay_path))


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
                    source = _create_replay_fallback()

            else:
                source = _create_replay_fallback()

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
