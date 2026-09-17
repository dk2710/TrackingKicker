"""Read camera frames published by the standalone CameraProcess."""

from __future__ import annotations

import time
from typing import Optional

from ..Objects.Frames import Frame
from .Camera import Camera
try:
    from v3.camera_frame_ipc import DEFAULT_NAME, SharedFrameRing
except ModuleNotFoundError:  # Running with v3 as the working directory.
    from camera_frame_ipc import DEFAULT_NAME, SharedFrameRing


class CameraProcessSource:
    """Tracker source backed by the CameraProcess shared-memory ring.

    The tracker never opens pypylon and never claims the USB camera.  Frames
    are consumed in sequence so frames skipped by the processing queue can be
    retained by the existing retro-analysis path.
    """

    def __init__(self, cfg: Camera):
        if cfg is None:
            raise ValueError("Eine Kamerakonfiguration ist erforderlich")

        self.cfg = cfg
        self.fps = float(getattr(cfg, "force_fps", 180) or 180)
        name = str(getattr(cfg, "frame_ipc_name", DEFAULT_NAME) or DEFAULT_NAME)
        self.ring = SharedFrameRing.open(name)
        expected = (int(cfg.height), int(cfg.width))
        actual = (self.ring.height, self.ring.width)
        if actual != expected:
            self.ring.close()
            raise RuntimeError(
                f"CameraProcess-Framegroesse {actual} passt nicht zur Tracker-Konfiguration "
                f"{expected}"
            )

        # Attach at the live edge. A tracker restart must not replay old frames
        # that were produced before this instance started.
        self._last_sequence = self.ring.latest_sequence()
        self._initialized = False
        self._closed = False
        self._read_count = 0
        self._read_failures = 0
        self._last_log_time = time.monotonic()
        self._last_frame_time = 0.0
        self._measured_fps = self.fps
        self._dropped_frames = 0

        print(
            f"[CAMERA-PROCESS] Verbunden: {name} "
            f"({self.ring.width}x{self.ring.height}, {self.ring.slots} Slots)"
        )

    def read(self):
        if self._closed:
            return False, None

        # Let the first call start at the newest available frame even when the
        # producer had already filled the ring before the tracker connected.
        if not self._initialized:
            self._last_sequence = self.ring.latest_sequence()
            self._initialized = True

        if getattr(self.cfg, "grab_strategy_latest_only", False):
            item = self.ring.read_latest(self._last_sequence)
        else:
            item = self.ring.read_next(self._last_sequence)
        if item is None:
            self._read_failures += 1
            self._report()
            return False, None

        sequence, timestamp, image = item
        if sequence > self._last_sequence + 1:
            self._dropped_frames += sequence - self._last_sequence - 1
        self._last_sequence = sequence

        frame = Frame()
        frame.frame = image
        frame.timestamp = float(timestamp)
        self._read_count += 1
        self._last_frame_time = time.monotonic()
        self._report()
        return True, frame

    def _report(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_log_time
        if elapsed < 1.0:
            return
        self._measured_fps = self._read_count / elapsed
        print(
            f"[CAMERA-PROCESS] geliefert={self._read_count} "
            f"fps={self._measured_fps:.1f} "
            f"fehlgeschlagen={self._read_failures} "
            f"uebersprungen={self._dropped_frames}"
        )
        self._read_count = 0
        self._read_failures = 0
        self._last_log_time = now

    def showFPS(self) -> str:
        return f"{self._measured_fps:.2f}"

    def status(self):
        return {
            "read_failures": int(self._read_failures),
            "dropped_frames": int(self._dropped_frames),
            "strategy": (
                "LatestImageOnly"
                if getattr(self.cfg, "grab_strategy_latest_only", False)
                else "OneByOne"
            ),
        }

    def reset(self) -> None:
        """Keep the source interface; the CameraProcess has no replay cursor."""

    def get_temperature(self) -> Optional[str]:
        return None

    def getTemperatur(self, camera=None) -> Optional[str]:
        return self.get_temperature()

    def release(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.ring.close()
