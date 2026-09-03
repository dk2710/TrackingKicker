"""Lens distortion correction for camera images."""

from dataclasses import dataclass

import cv2
import numpy as np

from .Camera import Camera


@dataclass
class DistortionParams:
    k1: float = 0.00
    k2: float = 0.05
    k3: float = 0.00
    p1: float = 0.00
    p2: float = 0.00
    scale: float = 1.00


class DistortionCorrector:
    def __init__(self, camera: Camera):
        self.camera = camera
        self._map_x: np.ndarray | None = None
        self._map_y: np.ndarray | None = None
        self._last_key: tuple | None = None

    @staticmethod
    def _compute_map(
        w: int,
        h: int,
        k1: float,
        k2: float,
        k3: float,
        p1: float,
        p2: float,
        scale: float,
    ):
        cx, cy = w / 2.0, h / 2.0
        ys, xs = np.indices((h, w), np.float32)
        x = (xs - cx) / cx
        y = (ys - cy) / cy
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        x_t = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        y_t = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x2 = (x * radial + x_t) * scale
        y2 = (y * radial + y_t) * scale
        return (
            (x2 * cx + cx).astype(np.float32),
            (y2 * cy + cy).astype(np.float32),
        )

    def apply(self, frame: np.ndarray, *, interpolation=cv2.INTER_LINEAR) -> np.ndarray:
        if frame is None:
            return frame

        h, w = frame.shape[:2]
        c = self.camera
        key = (w, h, c.k1, c.k2, c.k3, c.p1, c.p2, c.scale)
        if key != self._last_key or self._map_x is None:
            self._map_x, self._map_y = self._compute_map(
                w, h, c.k1, c.k2, c.k3, c.p1, c.p2, c.scale
            )
            self._last_key = key

        return cv2.remap(frame, self._map_x, self._map_y, interpolation)

