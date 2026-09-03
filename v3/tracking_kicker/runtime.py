"""Application-wide runtime configuration and replay discovery."""

import os
from pathlib import Path

try:
    import fcntl as _fcntl
except ImportError:  # Windows/deployments without POSIX file locking
    _fcntl = None

from .camera import RawReplaySource
from .config import AppConfig
from .model.VisionConfig import VisionConfig


vision = VisionConfig.load()
config = AppConfig.load()

VIDEO_PATH = config.videos_dir / "test.npz"
ANALYSIS_VIDEO_PATH = (Path(__file__).resolve().parents[1] / "videos").resolve()
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

GOAL_VIDEO_DURATION_SECONDS = 3
GOAL_PRE_NPZ_DURATION_SECONDS = 1.0
GOAL_VIDEO_FRAME_LIMIT = 60
GOAL_VIDEO_FPS = GOAL_VIDEO_FRAME_LIMIT / GOAL_VIDEO_DURATION_SECONDS
GOAL_VIDEO_MAX_WIDTH = 640

RETRO_TRIGGER_NO_BALL_FRAMES = 80
RETRO_MAX_REANALYZE_FRAMES = RETRO_TRIGGER_NO_BALL_FRAMES
RETRO_BUFFER_MAX_FRAMES = 160
RETRO_BALL_LOSS_SECONDS = 0.5
RETRO_REARM_WINDOW_SECONDS = 0.5
RETRO_REARM_MIN_BALL_DETECTIONS = 3

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


def find_replay_video() -> Path:
    """Find a usable NPZ replay for camera-less operation."""
    candidates = []
    if VIDEO_PATH.exists():
        candidates.append(VIDEO_PATH)

    replay_directories = [config.videos_dir, ANALYSIS_VIDEO_PATH]
    if config.debug_video:
        configured_path = Path(config.debug_video)
        if not configured_path.is_absolute():
            candidates.extend(directory / configured_path for directory in replay_directories)
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

    if VIDEO_PATH.resolve() in existing:
        return VIDEO_PATH.resolve()
    return max(existing, key=lambda path: path.stat().st_mtime_ns)


def create_replay_fallback():
    replay_path = find_replay_video()
    print(f"[REPLAY] Fallback-Quelle: {replay_path}")
    return RawReplaySource(str(replay_path))
