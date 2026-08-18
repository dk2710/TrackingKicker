import cv2
import numpy as np
import math
from typing import Optional, Tuple
from v3.tracking_kicker.Objects.Frames import Frame

class BallDetector:
    def __init__(
        self,
        hsv_low: Tuple[int, int, int] = (100, 110, 90),
        hsv_high: Tuple[int, int, int] = (112, 255, 255),
        area_min: float = 60.0,
        area_max: float = 12000.0,
        debug: bool = False,
        detection_scale: float = 0.5,
    ):
        self.hsv_low = hsv_low
        self.hsv_high = hsv_high
        self.area_min = area_min
        self.area_max = area_max
        self.debug = debug
        self.detection_scale = detection_scale

        self.expected_r: Optional[float] = None
        self._last_xy: Optional[Tuple[int, int]] = None
        self._miss = 0
        self.max_hold_frames = 2



    # ---------------------------------------------------------
    # Mask
    # ---------------------------------------------------------
    def _mask_ball(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BayerRG2BGR)
        else:
            frame_bgr = frame

        hsv = cv2.cvtColor(cv2.GaussianBlur(frame_bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_low, self.hsv_high)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        if self.debug:
            cv2.imshow("Ball mask", mask)

        return mask

    # ---------------------------------------------------------
    # Geometry
    # ---------------------------------------------------------
    @staticmethod
    def _circularity(c) -> float:
        a = cv2.contourArea(c)
        p = cv2.arcLength(c, True)
        return 4.0 * math.pi * a / (p * p) if p > 1e-6 else 0.0

    @staticmethod
    def _solidity(c) -> float:
        a = cv2.contourArea(c)
        if a <= 1e-6:
            return 0.0
        hull = cv2.convexHull(c)
        ha = cv2.contourArea(hull)
        return a / ha if ha > 1e-6 else 0.0

    def _score(self, c, exp_r: Optional[float]):
        a = cv2.contourArea(c)
        (x, y), r = cv2.minEnclosingCircle(c)

        circ = self._circularity(c)
        sol = self._solidity(c)

        r_score = 0.0
        if exp_r:
            rel = abs(r - exp_r) / exp_r
            r_score = -rel

        score = (2.2 * circ) + (1.2 * sol) + (0.6 * r_score) + (0.1 * math.log(max(a, 1.0)))
        return score, (int(x), int(y), int(r))

    # ---------------------------------------------------------
    # Detection
    # ---------------------------------------------------------
    def detect(self, frame: Frame):
        image = frame.frame
        scale = self.detection_scale

        # Die Vollbild-Erkennung ist auf dem Orange Pi sehr teuer. Die
        # Detektion erfolgt halbaufgeloest; die Koordinaten werden unten
        # wieder auf die Originalaufloesung hochgerechnet.
        if 0.0 < scale < 1.0:
            image = cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
        else:
            scale = 1.0

        mask = self._mask_ball(image)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not cnts:
            frame.ball = None
            return None

        candidates = []

        for c in cnts:
            (x, y), r = cv2.minEnclosingCircle(c)

            # Radius-Filter
            if not (5 * scale < r < 25 * scale):
                continue

            # Circularity
            circ = self._circularity(c)
            if circ < 0.5:
                continue

            # Solidity
            sol = self._solidity(c)
            if sol < 0.85:
                continue

            candidates.append((c, (x, y, r)))

        if not candidates:
            frame.ball = None
            return None

        _, (x, y, r) = max(candidates, key=lambda t: cv2.contourArea(t[0]))

        # Hier speicherst du ALLES in einem Feld
        frame.ball = (
            int(x / scale),
            int(y / scale),
            int(r / scale),
        )

        return True




