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
        debug: bool = False,
    ):
        self.hsv_low = hsv_low
        self.hsv_high = hsv_high
        self.area_min = area_min
        self.area_max = area_max
        self.use_morph = use_morph
        self.debug = debug

        # aus "guten" Detektionen gelernter Radius (keine Vorhersage, nur Stabilisierung)
        self.expected_r: Optional[float] = None

    def _mask_ball(self, frame_bgr: np.ndarray) -> np.ndarray:
        # leichtes Blur hilft gegen "löchrige" Maske bei Motion Blur / Rauschen
        blur = cv2.GaussianBlur(frame_bgr, (5, 5), 0)

        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_low, self.hsv_high)

        if self.use_morph:
            # stärkeres Closing verbindet gelbe Fragmente (z.B. hinter Stange)
            k_close = np.ones((9, 9), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)

            # leichtes Opening gegen Einzelpixel
            k_open = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)

            # optional: 1x Dilate, um kleine Lücken im Ball zu schließen
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

        return mask

    def mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        m = self._mask_ball(frame_bgr)
        if self.debug:
            cv2.imshow("Ball mask (FULL)", m)
        return m

    @staticmethod
    def _circularity(contour) -> float:
        a = cv2.contourArea(contour)
        p = cv2.arcLength(contour, True)
        if p <= 1e-6:
            return 0.0
        return float(4.0 * math.pi * a / (p * p))

    def _is_reliable(self, contour, r: float) -> bool:
        # Heuristik: bei Fragmenten ist Circularity oft klein
        circ = self._circularity(contour)
        return (circ >= 0.45) and (r >= 2.0)

    def detect(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int]]:
        mask = self._mask_ball(frame_bgr)

        if self.debug:
            cv2.imshow("Ball mask (FULL)", mask)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        # 1) Kandidat: größte passende Kontur
        best_c = None
        best_area = 0.0
        for c in cnts:
            a = cv2.contourArea(c)
            if self.area_min < a < self.area_max and a > best_area:
                best_area = a
                best_c = c

        if best_c is None:
            return None

        (x, y), r = cv2.minEnclosingCircle(best_c)
        x_i, y_i, r_i = int(x), int(y), int(r)

        reliable = self._is_reliable(best_c, r)

        # "guten" Radius merken (glätten)
        if reliable:
            if self.expected_r is None:
                self.expected_r = float(r)
            else:
                self.expected_r = 0.9 * self.expected_r + 0.1 * float(r)

            return (x_i, y_i, max(2, r_i))

        # 2) Unreliable (Fragment): Kreis-Recovery via Hough in ROI um die Maske
        #    -> nutzt reale Bildkanten, keine Vorhersage
        if self.expected_r is not None:
            exp_r = float(self.expected_r)

            # ROI aus BoundingBox der Kontur (+padding)
            bx, by, bw, bh = cv2.boundingRect(best_c)
            pad = int(max(15, exp_r * 1.5))
            H, W = frame_bgr.shape[:2]
            x0 = max(0, bx - pad)
            y0 = max(0, by - pad)
            x1 = min(W, bx + bw + pad)
            y1 = min(H, by + bh + pad)

            roi = frame_bgr[y0:y1, x0:x1]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 1.5)

            minR = max(2, int(exp_r * 0.7))
            maxR = max(minR + 2, int(exp_r * 1.3))

            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=max(10, int(exp_r)),
                param1=120,
                param2=18,     # kleiner = sensitiver; bei False-Positives erhöhen (20-30)
                minRadius=minR,
                maxRadius=maxR
            )

            if circles is not None:
                c0 = circles[0][0]  # bester
                cx, cy, cr = float(c0[0]), float(c0[1]), float(c0[2])
                return (int(round(cx + x0)), int(round(cy + y0)), int(round(cr)))

            # Fallback: Fragment-Zentrum, aber Radius stabilisiert (keine Mini-Kreise mehr)
            return (x_i, y_i, int(round(exp_r)))

        # Ohne expected_r: nur Fragment zurückgeben
        return (x_i, y_i, max(2, r_i))
