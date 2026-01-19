import cv2
import numpy as np
import math
from typing import Optional, Tuple


class KalmanBallTracker:
    """
    Trackt den Ballmittelpunkt (x,y) mit konstantem Geschwindigkeitsmodell.
    Radius wird als "letzter stabiler Radius" geführt und NICHT von Teil-Fragmenten übernommen.
    """

    def __init__(
        self,
        fps: float,
        gate_px: float = 180.0,        # wie weit Messung von Prediction abweichen darf
        min_r_ratio: float = 0.65,     # Messradius < min_r_ratio * last_r => "teilweise sichtbar" => nicht korrigieren
        reacquire_px: float = 450.0    # wenn Messung SEHR weit weg ist, hart neu initialisieren
    ):
        self.dt = 1.0 / max(1.0, float(fps))
        self.gate_px = float(gate_px)
        self.reacquire_px = float(reacquire_px)
        self.min_r_ratio = float(min_r_ratio)

        self.kf = cv2.KalmanFilter(4, 2)  # [x,y,vx,vy] -> [x,y]

        self.kf.transitionMatrix = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # Tuning (glatt, aber nicht träge)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 5e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5e-1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0

        self.initialized = False
        self.last_r: int = 12

    def _dist(self, ax: float, ay: float, bx: float, by: float) -> float:
        return math.hypot(ax - bx, ay - by)

    def update(self, measurement: Optional[Tuple[int, int, int]]) -> Tuple[Optional[Tuple[int, int, int]], bool]:
        """
        measurement = (mx,my,mr) oder None
        returns (x,y,r), predicted_flag
        predicted_flag=True bedeutet: Mittelpunkt ist (hauptsächlich) Prediction (z.B. Occlusion)
        """

        pred = self.kf.predict()
        px, py = float(pred[0]), float(pred[1])

        # ohne Initialisierung und ohne Messung: nix
        if measurement is None and not self.initialized:
            return None, True

        # Messung vorhanden
        if measurement is not None:
            mx, my, mr = measurement

            if not self.initialized:
                self.kf.statePost = np.array([[mx], [my], [0], [0]], dtype=np.float32)
                self.initialized = True
                self.last_r = int(mr)
                return (mx, my, self.last_r), False

            dist = self._dist(mx, my, px, py)

            # Occlusion-Heuristik: Radius viel kleiner als normal => Fragment => Zentrum ist unzuverlässig
            r_unreliable = mr < (self.min_r_ratio * self.last_r)

            # Wenn Messung extrem weit weg ist (z.B. Detektor springt): hart auf Messung resetten
            if dist > self.reacquire_px and not r_unreliable:
                self.kf.statePost = np.array([[mx], [my], [0], [0]], dtype=np.float32)
                # Radius nur updaten, wenn plausibel
                self.last_r = int(round(0.85 * self.last_r + 0.15 * mr))
                return (mx, my, self.last_r), False

            # Wenn Messung nahe genug UND nicht "Fragment": normal korrigieren
            if dist <= self.gate_px and not r_unreliable:
                z = np.array([[mx], [my]], dtype=np.float32)
                self.kf.correct(z)

                # Radius nur leicht glätten
                self.last_r = int(round(0.85 * self.last_r + 0.15 * mr))

                sx, sy = float(self.kf.statePost[0]), float(self.kf.statePost[1])
                return (int(round(sx)), int(round(sy)), self.last_r), False

            # Messung vorhanden, aber unzuverlässig (Fragment) ODER zu weit weg => nur Prediction verwenden
            if self.initialized:
                return (int(round(px)), int(round(py)), int(self.last_r)), True

        # Keine Messung => Prediction
        return (int(round(px)), int(round(py)), int(self.last_r)), True
