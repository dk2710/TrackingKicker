"""Latest-frame handoff for camera and website threads in one process."""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np


class LatestFrameBuffer:
    """Non-blocking in-process mailbox that keeps only the newest frame.

    The producer owns the captured numpy array and never waits for the web
    preview. Consumers receive the immutable frame object by reference; the
    preview thread copies it only when it starts image processing.
    """

    def __init__(self, width: int, height: int) -> None:
        self._lock = threading.Lock()
        self.width = int(width)
        self.height = int(height)
        self._sequence = 0
        self._timestamp = 0.0
        self._latest: Optional[np.ndarray] = None
        self._published = 0
        self._replaced = 0

    def resize(self, width: int, height: int) -> None:
        with self._lock:
            self.width = int(width)
            self.height = int(height)
            self._latest = None
            self._timestamp = 0.0

    def publish(self, image: np.ndarray, timestamp: Optional[float] = None) -> int:
        array = np.asarray(image)
        expected = (self.height, self.width)
        if array.shape != expected or array.dtype != np.uint8:
            raise ValueError(
                f"Erwartet uint8-Frame {expected}, erhalten {array.shape} {array.dtype}"
            )
        with self._lock:
            if self._latest is not None:
                self._replaced += 1
            self._sequence += 1
            self._latest = array
            self._timestamp = float(timestamp or time.monotonic())
            self._published += 1
            return self._sequence

    def latest_sequence(self) -> int:
        with self._lock:
            return self._sequence

    def read_latest(self, last_sequence: int):
        with self._lock:
            if self._latest is None or self._sequence <= int(last_sequence):
                return None
            return self._sequence, self._timestamp, self._latest

    def stats(self) -> dict:
        with self._lock:
            return {
                "mode": "in_process_latest_frame",
                "available": self._latest is not None,
                "width": self.width,
                "height": self.height,
                "sequence": self._sequence,
                "frames_published": self._published,
                "frames_replaced": self._replaced,
            }
