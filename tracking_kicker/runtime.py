"""Application-wide runtime configuration."""

import os
from dataclasses import replace
from pathlib import Path

try:
    import fcntl as _fcntl
except ImportError:  # Windows/deployments without POSIX file locking
    _fcntl = None

from .camera.CameraProcessSource import CameraProcessSource
from .config import AppConfig
from .model.VisionConfig import VisionConfig


vision = VisionConfig.load()
config = AppConfig.load()

ANALYSIS_VIDEO_PATH = config.videos_dir.resolve()
ANALYSIS_VIDEO_PATH.mkdir(parents=True, exist_ok=True)

# Remove only obsolete files created by the former segment encoder.
for old_goal_file in ANALYSIS_VIDEO_PATH.iterdir():
    if (
        old_goal_file.is_file()
        and old_goal_file.name != "goal_latest.webm"
        and (
            old_goal_file.name.startswith("goal_ring_")
            or old_goal_file.name.startswith("goal_")
        )
        and old_goal_file.suffix in {".webm", ".csv"}
    ):
        try:
            old_goal_file.unlink()
        except OSError as cleanup_error:
            print(
                f"[GOAL] Alte Videodatei nicht entfernbar: "
                f"{old_goal_file}: {cleanup_error}"
            )


headless_env = os.environ.get("HEADLESS")
no_linux_display = (
    os.name == "posix"
    and not os.environ.get("DISPLAY")
    and not os.environ.get("WAYLAND_DISPLAY")
)
HEADLESS = (
    config.headless if headless_env is None else headless_env == "1"
) or no_linux_display

GOAL_VIDEO_DURATION_SECONDS = 3.0
GOAL_ANALYSIS_DURATION_SECONDS = 1.0
GOAL_VIDEO_FPS = 30.0
GOAL_VIDEO_FRAME_LIMIT = int(GOAL_VIDEO_DURATION_SECONDS * GOAL_VIDEO_FPS)
GOAL_VIDEO_MAX_WIDTH = 640

DIGIKICKER_POST_DEFAULT_URL = "https://localhost/goal"
DIGIKICKER_POST_TIMEOUT_SECONDS = 2.0

_app_lock_handle = None


def acquire_app_lock() -> bool:
    """Prevent two processes from opening the same Basler camera."""
    global _app_lock_handle
    if _fcntl is None:
        return True
    try:
        _app_lock_handle = open("/tmp/tracking_kicker.lock", "w")
        _fcntl.flock(_app_lock_handle.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        _app_lock_handle.write(str(os.getpid()))
        _app_lock_handle.flush()
        return True
    except (BlockingIOError, OSError):
        print("[APP] Bereits eine TrackingKicker-Instanz aktiv; zweiter Start beendet.")
        return False


def create_replay_fallback():
    if config.camera is None:
        raise RuntimeError(
            "Fuer den Raw Replay Process wird eine Kamera-Konfiguration "
            "mit Framegroesse benoetigt"
        )
    print(
        "[REPLAY] Erwarte den separaten TrackingKicker-CameraProcess."
    )
    replay_camera = replace(
        config.camera,
        source="camera_process",
        frame_ipc_name=config.replay_ipc_name,
        frame_ipc_slots=config.replay_ipc_slots,
    )
    return CameraProcessSource(replay_camera)
