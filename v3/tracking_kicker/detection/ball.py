import cv2
import numpy as np
import math
from typing import Optional, Tuple


class BallDetector:
    def __init__(
        self,
        hsv_low: Tuple[int, int, int] = (18, 80, 110),
        hsv_high: Tuple[int, int, int] = (38, 255, 255),
        area_min: float = 80.0,
        area_max: float = 8000.0,
        use_morph: bool = True,
        use_tracking_roi: bool = True,
        search_radius: int = 180,
        debug: bool = False
    ):
        self.hsv_low = hsv_low
        self.hsv_high = hsv_high
        self.area_min = area_min
        self.area_max = area_max
        self.use_morph = use_morph
        self.use_tracking_roi = use_tracking_roi
        self.search_radius = search_radius
        self.debug = debug

        self._prev_xy: Optional[Tuple[int, int]] = None
        self.expected_r: Optional[float] = None  # wird aus "guten" Frames gelernt
        self.last_reliable: bool = False         # <- wichtig für den Tracker

    def _mask_ball(self, frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_low, self.hsv_high)

        if self.use_morph:
            k = np.ones((7, 7), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

        return mask

    def _choose_best(self, mask: np.ndarray, x_off: int, y_off: int) -> Optional[Tuple[int, int, int]]:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_score = -1.0
        best_reliable = False
        best_r = None

        for c in cnts:
            a = cv2.contourArea(c)
            if not (self.area_min < a < self.area_max):
                continue

            p = cv2.arcLength(c, True)
            if p <= 0:
                continue

            circularity = 4.0 * math.pi * a / (p * p)

            hull = cv2.convexHull(c)
            (x, y), r0 = cv2.minEnclosingCircle(hull)
            if r0 <= 1:
                continue

            fill = a / (math.pi * (r0 * r0) + 1e-6)

            # "zu un-rund" oder zu dünn => eher Fragment / falscher Blob
            reliable = (circularity >= 0.45) and (fill >= 0.30)

            # radius Nähe zur erwarteten Größe (falls gelernt)
            r_penalty = 1.0
            if self.expected_r is not None:
                rel = abs(r0 - self.expected_r) / max(self.expected_r, 1e-6)
                r_penalty = math.exp(-3.0 * rel)  # näher = besser

            score = a * max(0.0, min(circularity, 1.0)) * max(0.0, min(fill, 1.5)) * r_penalty

            if score > best_score:
                best_score = score
                best = (int(x) + x_off, int(y) + y_off, int(r0))
                best_reliable = reliable
                best_r = float(r0)

        # Flag setzen + erwarteten Radius nur bei "guten" Frames updaten
        self.last_reliable = bool(best_reliable)
        if best is not None and best_reliable and best_r is not None:
            if self.expected_r is None:
                self.expected_r = best_r
            else:
                self.expected_r = 0.9 * self.expected_r + 0.1 * best_r

        # Wenn Fragment, aber expected_r bekannt: gebe trotzdem den "vollen" Radius zurück
        if best is not None and (not best_reliable) and (self.expected_r is not None):
            x, y, _ = best
            best = (x, y, int(round(self.expected_r)))

        return best

    def detect(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int]]:
        H, W = frame_bgr.shape[:2]

        # 1) ROI um letzte Position
        if self.use_tracking_roi and self._prev_xy is not None:
            cx, cy = self._prev_xy
            R = self.search_radius
            x0 = max(0, cx - R)
            y0 = max(0, cy - R)
            x1 = min(W, cx + R)
            y1 = min(H, cy + R)

            roi = frame_bgr[y0:y1, x0:x1]
            mask = self._mask_ball(roi)
            if self.debug:
                cv2.imshow("Ball mask (ROI)", mask)

            best = self._choose_best(mask, x_off=x0, y_off=y0)
            if best is not None:
                self._prev_xy = (best[0], best[1])
                return best

        # 2) Fallback: Full frame
        mask_full = self._mask_ball(frame_bgr)
        if self.debug:
            cv2.imshow("Ball mask (FULL)", mask_full)

        best = self._choose_best(mask_full, x_off=0, y_off=0)
        if best is not None:
            self._prev_xy = (best[0], best[1])
        return best

    def mask(self, frame_bgr):
        # liefert die gleiche Ball-Maske wie für detect()
        return self._mask_ball(frame_bgr)

