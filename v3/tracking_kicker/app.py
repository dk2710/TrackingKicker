"""Application orchestrator for TrackingKicker.

The domain implementations live in their packages:

* :mod:`detection.ball` owns ball detection
* :mod:`camera` owns Basler/replay sources and pixel conversion
* :mod:`analysis` owns goal and speed analysis
* :mod:`calibration` owns calibration
* :mod:`comms.http_api` owns HTTP endpoints
* :mod:`ui` owns OpenCV interaction
* :mod:`workers` owns the camera/processing worker loops
* :mod:`goal_media` owns goal notifications and video generation

This module intentionally only wires those pieces together and manages the
application lifecycle.
"""

import os
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, Optional

import cv2
from flask import Flask

from .analysis import KickerAnalyzer
from .calibration.workflow import CalibrationWorkflowMixin
from .comms.http_api import ApiMixin
from .detection import BallDetector
from .camera.distortion import DistortionCorrector
from .camera.factory import create_camera_with_retry
from .goal_media import GoalMediaMixin
from .logging_support import configure_runtime_logging
from .runtime import (
    ANALYSIS_VIDEO_PATH,
    DIGIKICKER_POST_DEFAULT_URL,
    DIGIKICKER_POST_TIMEOUT_SECONDS,
    GOAL_PRE_NPZ_DURATION_SECONDS,
    GOAL_VIDEO_DURATION_SECONDS,
    GOAL_VIDEO_FRAME_LIMIT,
    GOAL_VIDEO_FPS,
    GOAL_VIDEO_MAX_WIDTH,
    HEADLESS,
    RETRO_BUFFER_MAX_FRAMES,
    RETRO_MAX_REANALYZE_FRAMES,
    RETRO_TRIGGER_NO_BALL_FRAMES,
    acquire_app_lock,
    config,
    create_replay_fallback,
    vision,
)
from .ui.opencv_gui import destroy_all, imshow, wait_key
from .ui.opencv_gui import (
    destroy_all as gui_destroy_all,
    destroy_window as gui_destroy_window,
    imshow as gui_imshow,
    named_window as gui_named_window,
    wait_key as gui_wait_key,
)
from .workers import WorkerMixin


# Backwards-compatible aliases for code that imported the old private names.
Analyse_Video_Path = ANALYSIS_VIDEO_PATH
_acquire_app_lock = acquire_app_lock
_create_camera_with_retry = create_camera_with_retry
_create_replay_fallback = create_replay_fallback


class KickerApp(
    WorkerMixin,
    GoalMediaMixin,
    ApiMixin,
    CalibrationWorkflowMixin,
):
    """Coordinate application state and the independent runtime services."""

    def __init__(self, source):
        self.video = source
        self.cfg: Optional[Dict[str, Any]] = None
        self.dist = DistortionCorrector(config.camera)
        print(f"📁 Verzerrung config: {config.camera}")
        cv2.setUseOptimized(True)

        # Queues shared by the camera, processing and recording workers.
        self.frame_queue = Queue(maxsize=2)
        self.display_queue = Queue(maxsize=1)
        self.recording_queue = Queue()
        self.raw_queue = Queue(maxsize=2)
        self.processed_queue = Queue(maxsize=2)

        self.detection_state_lock = threading.Lock()
        self.last_detection = {}
        self.processing_fps = 0.0

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

        self.stop_event = threading.Event()
        self.record_event = threading.Event()

        # Raw replay buffer and bounded retro-analysis buffer.
        self.replay_buffer = deque(maxlen=int(240 * config.replay_seconds))
        self.replay_lock = threading.Lock()
        self.snapshot_lock = threading.Lock()
        self.snapshot_generation = 0
        self.snapshot_frames = []
        self.calibration_lock = threading.RLock()
        self.history_frames = deque()
        self.skipped_frame_lock = threading.Lock()
        self.skipped_frame_buffer = deque(maxlen=RETRO_BUFFER_MAX_FRAMES)
        self._frame_sequence = 0
        self._last_visible_frame_id = None
        self._last_retro_frame_id = None
        self._no_ball_live_frames = 0
        self._retro_analysis_count = 0

        # Last goal result and digiKicker notification settings.
        self.last_analysis_lock = threading.Lock()
        self._last_posted_goal_event_id = 0
        self.digikicker_post_url = os.environ.get(
            "DIGIKICKER_POST_URL", DIGIKICKER_POST_DEFAULT_URL
        ).strip()
        self.digikicker_post_verify_tls = os.environ.get(
            "DIGIKICKER_POST_VERIFY_TLS", "0"
        ).lower() in {"1", "true", "yes", "on"}
        try:
            self.digikicker_post_timeout = float(
                os.environ.get(
                    "DIGIKICKER_POST_TIMEOUT",
                    str(DIGIKICKER_POST_TIMEOUT_SECONDS),
                )
            )
        except ValueError:
            self.digikicker_post_timeout = DIGIKICKER_POST_TIMEOUT_SECONDS

        self.api = Flask(__name__)
        self.last_analysis = {}
        self.api.add_url_rule("/", "dashboard", self.api_dashboard, methods=["GET"])
        self.api.add_url_rule(
            "/live.mjpg",
            "live_stream",
            self.api_live_stream,
            methods=["GET"],
        )
        self.api.add_url_rule(
            "/api/status",
            "tracking_status",
            self.api_status,
            methods=["GET"],
        )
        self.api.add_url_rule(
            "/api/logs",
            "runtime_logs",
            self.api_logs,
            methods=["GET"],
        )
        self.api.add_url_rule(
            "/status",
            "status_alias",
            self.api_status,
            methods=["GET"],
        )
        self.api.add_url_rule(
            "/api/parameters",
            "parameters",
            self.api_parameters,
            methods=["GET", "POST"],
        )
        self.api.add_url_rule(
            "/api/snapshot",
            "snapshot",
            self.api_snapshot,
            methods=["GET", "POST"],
        )
        self.api.add_url_rule(
            "/api/snapshot/frame",
            "snapshot_frame",
            self.api_snapshot_frame,
            methods=["GET"],
        )
        self.api.add_url_rule(
            "/api/calibration",
            "calibration",
            self.api_calibration,
            methods=["GET", "POST"],
        )
        self.api.add_url_rule("/test", "test", self.api_test, methods=["GET"])
        self.api.add_url_rule("/goal", "goal", self.api_goal, methods=["GET"])
        self.api.add_url_rule(
            "/goal/video/<path:filename>",
            "goal_video",
            self.api_goal_video,
            methods=["GET"],
        )

        self.detector = BallDetector.from_config(config, vision.field)
        self.snapshot_detector = BallDetector.from_config(config, vision.field)
        self.ball_detection_enabled = bool(
            getattr(config, "ball_detection_enabled", False)
        )
        print(
            "Ballerkennung: "
            + ("aktiv" if self.ball_detection_enabled else "deaktiviert")
        )
        self.analyzer = KickerAnalyzer(
            goals_cfg=vision.goals,
            field_cfg=vision.field,
        )

    def _request_shutdown(self, signum=None, _frame=None):
        signal_name = signal.Signals(signum).name if signum else "shutdown"
        print(f"\n🛑 Stop angefordert ({signal_name})")
        self.stop_event.set()

    def run(self) -> bool:
        self._restart_requested = False
        try:
            # Signals are delivered in the main thread, including headless mode.
            signal.signal(signal.SIGINT, self._request_shutdown)
            if hasattr(signal, "SIGTERM"):
                signal.signal(signal.SIGTERM, self._request_shutdown)

            self.stop_event.clear()
            self.record_event.clear()
            self._load_or_calibrate()
            assert self.cfg is not None

            self.t_api = threading.Thread(target=self.api_loop, daemon=True)
            self.t_cam = threading.Thread(target=self.camera_loop, daemon=True)
            self.t_proc = threading.Thread(target=self.processing_loop, daemon=True)
            self.t_rec = threading.Thread(target=self.recording_loop, daemon=True)

            self.t_cam.start()
            self.t_proc.start()
            self.t_rec.start()
            self.t_api.start()

            print("⌨️ Taste 'r' = Aufnahme Start/Stop | ESC/q = Ende | P = Neustart")
            if HEADLESS:
                print("🟢 HEADLESS=1 → keine OpenCV-GUI")
                while not self.stop_event.is_set():
                    time.sleep(0.2)
            else:
                self._run_gui()

        except KeyboardInterrupt:
            print("🛑 Programm durch Benutzer beendet")
        except Exception as exc:
            print(f"❌ Unerwarteter Fehler in app.run(): {exc}")
        finally:
            self.stop_event.set()
            self.record_event.clear()
            # Let the camera worker leave RetrieveResult() before releasing
            # the source.  Basler USB cleanup is bounded inside BaslerSource;
            # it must never block application shutdown indefinitely.
            self._join_threads()
            try:
                self.video.release()
                print("✅ Kamera/Videoquelle freigegeben")
            except Exception as exc:
                print(f"⚠️ Videoquelle konnte nicht freigegeben werden: {exc}")
            try:
                destroy_all()
            except Exception:
                pass

        return self._restart_requested

    def _run_gui(self) -> None:
        while not self.stop_event.is_set():
            try:
                frame = self.display_queue.get(timeout=0.5)
            except Empty:
                continue

            imshow("Kicker Analyse", frame)
            key = wait_key(1)
            if key == ord("q"):
                print("KEY → Ende")
                break
            if key == ord("r"):
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
                self.stop_event.set()
                # The caller checks this flag through the local lifecycle path.
                self._restart_requested = True
                break

    def _join_threads(self):
        for thread in [self.t_cam, self.t_proc, self.t_rec, self.t_roll, self.t_api]:
            if thread is None:
                continue
            try:
                thread.join(timeout=0.3)
            except Exception:
                pass


def main() -> None:
    configure_runtime_logging()
    if not _acquire_app_lock():
        sys.exit(1)

    # Keep OpenCV from competing with pypylon/libusb for CPU time on the
    # Orange Pi.  The camera worker remains responsive while the processing
    # worker handles the latest frame.
    cv2.setNumThreads(1)

    while True:
        source = None
        restart_requested = False
        try:
            if config.use_camera:
                try:
                    source = _create_camera_with_retry(config.camera)
                    print("📷 Quelle: Basler Kamera")
                except Exception as exc:
                    print(f"❌ Kamera nicht verfügbar: {exc}")
                    print("➡️ Fallback auf den separaten Raw Replay Process")
                    source = _create_replay_fallback()
            else:
                source = _create_replay_fallback()

            app = KickerApp(source)
            restart_requested = app.run()
        except KeyboardInterrupt:
            print("🛑 Programm durch Benutzer beendet")
            break
        except Exception as exc:
            print(f"❌ Unerwarteter Fehler im Main-Block: {exc}")
        finally:
            if source is not None:
                try:
                    source.release()
                    print("✅ Kamera freigegeben")
                except Exception as exc:
                    print(f"⚠️ Fehler beim Freigeben der Kamera: {exc}")
            try:
                destroy_all()
            except Exception:
                pass
            time.sleep(1)

        if not restart_requested:
            break


if __name__ == "__main__":
    main()
