import cv2
import numpy as np
import math
from typing import Optional, Tuple


class BallDetector:
    def __init__(
        self,
        hsv_low: Tuple[int, int, int] = (100, 110, 90),    # etwas toleranter
        hsv_high: Tuple[int, int, int] = (112, 255, 255), # etwas toleranter (Gelb kann variieren)
        area_min: float = 60.0,
        area_max: float = 12000.0,
        use_morph: bool = True,
        debug: bool = False,
        use_hough_recovery: bool = False,
        use_blur: bool = True,
        fast_path: bool = False,
    ):
        self.hsv_low = hsv_low
        self.hsv_high = hsv_high
        self.area_min = area_min
        self.area_max = area_max
        self.use_morph = use_morph
        self.debug = debug
        self.use_hough_recovery = bool(use_hough_recovery)
        self.use_blur = bool(use_blur)
        self.fast_path = bool(fast_path)

        # gelernter Radius (Stabilisierung)
        self.expected_r: Optional[float] = None

        # Last good detection (für kurze Aussetzer)
        self._last_xy: Optional[Tuple[int, int]] = None
        self._miss = 0
        self.max_hold_frames = 2  # hält 1-2 Frames stabil, wenn Ball kurz "weg"

    # -----------------------------
    # Masking ----- Angucken
    # -----------------------------
    def _mask_ball(self, frame: np.ndarray) -> np.ndarray:
        # Nur wenn 1-Kanal (Bayer/Gray) -> demosaic
        if frame.ndim == 2:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BayerRG2BGR)  # ggf. anderes Muster
        elif frame.ndim == 3 and frame.shape[2] == 3:
            frame_bgr = frame  # ist schon BGR
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            raise ValueError(f"Unexpected frame shape: {frame.shape}")

        if self.use_blur:
            frame_bgr = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_low, self.hsv_high)

        if self.use_morph:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        return mask

    def mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        m = self._mask_ball(frame_bgr)
        if self.debug:
            cv2.imshow("Ball mask (FULL)", m)
        return m

    # -----------------------------
    # Geometry helpers
    # -----------------------------
    @staticmethod
    def _circularity(contour) -> float:
        a = cv2.contourArea(contour)
        p = cv2.arcLength(contour, True)
        if p <= 1e-6:
            return 0.0
        return float(4.0 * math.pi * a / (p * p))

    @staticmethod
    def _solidity(contour) -> float:
        a = cv2.contourArea(contour)
        if a <= 1e-6:
            return 0.0
        hull = cv2.convexHull(contour)
        ha = cv2.contourArea(hull)
        if ha <= 1e-6:
            return 0.0
        return float(a / ha)

    def _score_contour(self, contour, exp_r: Optional[float]) -> Tuple[float, Tuple[int,int,int]]:
        a = cv2.contourArea(contour)
        (x, y), r = cv2.minEnclosingCircle(contour)
        if r < 2:
            return -1e9, (0,0,0)

        circ = self._circularity(contour)
        sol = self._solidity(contour)

        # Radius-Nähe (wenn expected_r existiert)
        r_score = 0.0
        if exp_r is not None and exp_r > 1e-6:
            # relative Abweichung
            rel = abs(r - exp_r) / exp_r
            r_score = -rel  # kleiner rel = besser

        # Area-Liebling: nicht blind größte, aber leicht bevorzugt
        a_score = math.log(max(a, 1.0))

        # Gesamt: Circularity + Solidity dominieren, Radius hilft stabilisieren
        score = (2.2 * circ) + (1.2 * sol) + (0.6 * r_score) + (0.10 * a_score)

        return score, (int(round(x)), int(round(y)), int(round(r)))

    def _is_reliable(self, contour, r: float) -> bool:
        circ = self._circularity(contour)
        sol = self._solidity(contour)
        # zuverlässiger: Kreisig + nicht zu konkav
        return (circ >= 0.55) and (sol >= 0.85) and (r >= 2.0)

    # -----------------------------
    # Detection
    # -----------------------------
    def _detect_fast(self, mask: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Low-overhead component detector for the full-rate v4 path."""
        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
        if int(count) <= 1:
            return self._fast_hold()

        component_stats = stats[1:]
        areas = component_stats[:, cv2.CC_STAT_AREA].astype(np.float32)
        widths = component_stats[:, cv2.CC_STAT_WIDTH].astype(np.float32)
        heights = component_stats[:, cv2.CC_STAT_HEIGHT].astype(np.float32)
        box_area = np.maximum(1.0, widths * heights)
        radii = np.maximum(2.0, np.sqrt(areas / math.pi))
        valid = (
            (areas > float(self.area_min))
            & (areas < float(self.area_max))
            & ((areas / box_area) >= 0.18)
        )
        expected_r = self.expected_r
        if expected_r is not None:
            relative = np.abs(radii - float(expected_r)) / max(float(expected_r), 1e-6)
            valid &= relative <= 1.45
            scores = areas / (1.0 + relative)
        else:
            scores = areas
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size == 0:
            return self._fast_hold()

        selected = int(valid_indices[int(np.argmax(scores[valid_indices]))]) + 1
        x_i = int(round(float(centroids[selected][0])))
        y_i = int(round(float(centroids[selected][1])))
        radius_i = int(round(float(radii[selected - 1])))
        radius = float(radius_i)
        if self.expected_r is None:
            self.expected_r = radius
        else:
            self.expected_r = 0.85 * self.expected_r + 0.15 * radius
        self._last_xy = (x_i, y_i)
        self._miss = 0
        return (x_i, y_i, max(2, radius_i))

    def _fast_hold(self) -> Optional[Tuple[int, int, int]]:
        if (
            self._last_xy is not None
            and self._miss < self.max_hold_frames
            and self.expected_r is not None
        ):
            self._miss += 1
            return (
                self._last_xy[0],
                self._last_xy[1],
                int(round(self.expected_r)),
            )
        self._last_xy = None
        return None

    def detect(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int]]:
        mask = self._mask_ball(frame_bgr)

        if self.debug:
            cv2.imshow("Ball mask (FULL)", mask)

        if self.fast_path:
            return self._detect_fast(mask)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            # kurzer Hold (gegen flackern)
            if self._last_xy is not None and self._miss < self.max_hold_frames and self.expected_r is not None:
                self._miss += 1
                return (self._last_xy[0], self._last_xy[1], int(round(self.expected_r)))
            self._last_xy = None
            return None

        exp_r = self.expected_r

        # Kandidaten filtern & scoren
        best_score = -1e9
        best_c = None
        best_trip = None

        for c in cnts:
            a = cv2.contourArea(c)
            (x, y), r = cv2.minEnclosingCircle(c)

            # Grundfilter
            if not (self.area_min < a < self.area_max):
                continue

            # Größenfilter nach gelerntem Ball
            if self.expected_r is not None:
                if r < 0.65 * self.expected_r:
                    continue
                if r > 1.45 * self.expected_r:
                    continue

            score, trip = self._score_contour(c, exp_r)
            if score > best_score:
                best_score = score
                best_c = c
                best_trip = trip

        if best_c is None or best_trip is None:
            # Hold
            if self._last_xy is not None and self._miss < self.max_hold_frames and self.expected_r is not None:
                self._miss += 1
                return (self._last_xy[0], self._last_xy[1], int(round(self.expected_r)))
            self._last_xy = None
            return None

        x_i, y_i, r_i = best_trip
        (_, _), r_float = cv2.minEnclosingCircle(best_c)

        # 🔒 Initiale Ballgröße setzen
        if self.expected_r is None and r_float >= 5:
            self.expected_r = float(r_float)

        reliable = self._is_reliable(best_c, r_float)

        if reliable:
            # expected_r glätten
            if self.expected_r is None:
                self.expected_r = float(r_float)
            else:
                self.expected_r = 0.85 * self.expected_r + 0.15 * float(r_float)

            self._last_xy = (x_i, y_i)
            self._miss = 0
            return (x_i, y_i, max(2, int(round(r_float))))

        # Recovery: Wenn wir expected_r haben, nutze Hough in ROI (deine Idee, aber etwas stabiler)
        if self.expected_r is not None and self.use_hough_recovery:
            exp_r = float(self.expected_r)

            bx, by, bw, bh = cv2.boundingRect(best_c)
            pad = int(max(18, exp_r * 2.0))
            H, W = frame_bgr.shape[:2]
            x0 = max(0, bx - pad)
            y0 = max(0, by - pad)
            x1 = min(W, bx + bw + pad)
            y1 = min(H, by + bh + pad)

            roi = frame_bgr[y0:y1, x0:x1]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 1.5)

            minR = max(2, int(exp_r * 0.7))
            maxR = max(minR + 2, int(exp_r * 1.35))

            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=max(10, int(exp_r)),
                param1=120,
                param2=16,  # leicht sensibler
                minRadius=minR,
                maxRadius=maxR
            )

            if circles is not None:
                c0 = circles[0][0]
                cx, cy, cr = float(c0[0]), float(c0[1]), float(c0[2])
                x_f = int(round(cx + x0))
                y_f = int(round(cy + y0))
                r_f = int(round(cr))

                self._last_xy = (x_f, y_f)
                self._miss = 0
                return (x_f, y_f, r_f)

            # Fallback: konturzentrum + expected radius
            self._last_xy = (x_i, y_i)
            self._miss = 0
            return (x_i, y_i, int(round(exp_r)))

        # Ohne expected_r: nimm bestes Fragment
        self._last_xy = (x_i, y_i)
        self._miss = 0
        return (x_i, y_i, max(2, r_i))
