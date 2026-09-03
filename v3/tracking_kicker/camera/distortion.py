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
        # cv2.remap is faster and uses less memory with fixed-point maps than
        # with two full CV_32FC1 maps.  The public behavior stays identical;
        # only the internal representation of the already cached map changes.
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

    def _maps_for_shape(self, w: int, h: int):
        c = self.camera
        key = (w, h, c.k1, c.k2, c.k3, c.p1, c.p2, c.scale)
        if key != self._last_key or self._map_x is None:
            map_x, map_y = self._compute_map(
                w, h, c.k1, c.k2, c.k3, c.p1, c.p2, c.scale
            )
            # CV_16SC2 + CV_16UC1 is the fixed-point representation expected
            # by remap. It retains the subpixel interpolation fractions and
            # avoids recalculating map coordinates for every output pixel.
            self._map_x, self._map_y = cv2.convertMaps(
                map_x,
                map_y,
                cv2.CV_16SC2,
                nninterpolation=False,
            )
            self._last_key = key
        return self._map_x, self._map_y

    def apply(self, frame: np.ndarray, *, interpolation=cv2.INTER_LINEAR) -> np.ndarray:
        if frame is None:
            return frame

        h, w = frame.shape[:2]
        map_x, map_y = self._maps_for_shape(w, h)
        return cv2.remap(frame, map_x, map_y, interpolation)

    def apply_scaled(
        self,
        frame: np.ndarray,
        scale: float,
        *,
        interpolation=cv2.INTER_LINEAR,
    ) -> np.ndarray:
        """Correct a reduced image using the same normalized camera model.

        The live detector already works on a reduced image.  Resizing before
        remapping avoids doing a full-resolution remap when no display image
        is needed.  The detector scales its coordinates back to the original
        frame, so this does not change the coordinate or speed units.
        """
        if frame is None:
            return frame

        scale = float(scale)
        if not 0.0 < scale < 1.0:
            return self.apply(frame, interpolation=interpolation)

        resized = cv2.resize(
            frame,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        h, w = resized.shape[:2]
        map_x, map_y = self._maps_for_shape(w, h)
        return cv2.remap(resized, map_x, map_y, interpolation)
