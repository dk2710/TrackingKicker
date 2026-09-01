import cv2
import numpy as np
import math
import time
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
        expected_fps: float = 180.0,
        max_speed_kmh: float = 350.0,
        table_width_cm: float = 120.0,
    ):
        self.hsv_low = hsv_low
        self.hsv_high = hsv_high
        self.area_min = area_min
        self.area_max = area_max
        self.debug = debug
        self.detection_scale = detection_scale

        # Die Konturerkennung liefert pro Frame Kandidaten, verfolgt aber
        # nicht automatisch denselben Kandidaten weiter. Diese Zustandsdaten
        # verhindern, dass ein Sprung auf eine andere blaue Kontur spaeter
        # als Ballbewegung in die Geschwindigkeitsmessung eingeht.
        self.expected_fps = max(float(expected_fps), 1.0)
        self.nominal_dt = 1.0 / self.expected_fps
        # Die Basler laeuft mit 180 FPS. Kleinere Zeitabstaende stammen in
        # der Praxis aus doppelten/fehlerhaften Zeitstempeln und sind fuer
        # die Geschwindigkeit nicht belastbar.
        self.min_track_dt = 0.0025
        self.max_track_dt = 0.25
        self.max_speed_kmh = float(max_speed_kmh)
        self.table_width_cm = max(float(table_width_cm), 1.0)
        self._track_xy: Optional[Tuple[float, float]] = None
        self._track_radius: Optional[float] = None
        self._track_timestamp: Optional[float] = None
        self._track_velocity_px_s = (0.0, 0.0)
        self._track_misses = 0
        self.max_track_misses = 3

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

        # Diese Attribute werden auch bei einer reinen Positionsanalyse
        # gesetzt. Nur ball_track_valid wird vom Analyzer fuer die
        # Geschwindigkeitsmessung ausgewertet.
        frame.ball_track_valid = False
        frame.ball_track_reason = "no_candidate"
        frame.ball_track_distance_px = None

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
            self._track_misses += 1
            if self._track_misses > self.max_track_misses:
                self._reset_track()
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
            self._track_misses += 1
            if self._track_misses > self.max_track_misses:
                self._reset_track()
            return None

        # Alle Kandidaten in Originalkoordinaten umrechnen. Die groesste
        # Kontur bleibt der Fallback fuer die allgemeine Positionsanalyse.
        normalized = []
        for contour, (x, y, r) in candidates:
            normalized.append({
                "center": (float(x / scale), float(y / scale)),
                "radius": float(r / scale),
                "area": float(cv2.contourArea(contour)),
            })

        fallback = max(normalized, key=lambda item: item["area"])
        chosen = fallback
        track_valid = True
        track_reason = "seed"
        distance_px = None

        timestamp = float(getattr(frame, "timestamp", time.perf_counter()))
        if self._track_xy is not None and self._track_timestamp is not None:
            raw_dt = timestamp - self._track_timestamp
            temporal_valid = (
                self.min_track_dt <= raw_dt <= self.max_track_dt
            )
            effective_dt = raw_dt if temporal_valid else self.nominal_dt

            vx, vy = self._track_velocity_px_s
            predicted = (
                self._track_xy[0] + vx * effective_dt,
                self._track_xy[1] + vy * effective_dt,
            )

            # Derive the pixel gate from a deliberately generous physical
            # upper bound. A real fast shot remains possible, while a
            # candidate on a far-away contour is not silently connected.
            max_px_per_second = (
                self.max_speed_kmh / 0.036
                / self.table_width_cm
                * frame.frame.shape[1]
            )
            max_distance = max(
                45.0,
                max_px_per_second * max(effective_dt, self.nominal_dt) + 20.0,
            )

            distances = [
                (
                    math.hypot(
                        item["center"][0] - predicted[0],
                        item["center"][1] - predicted[1],
                    ),
                    item,
                )
                for item in normalized
            ]
            nearby = [entry for entry in distances if entry[0] <= max_distance]
            if nearby:
                # Naehe zur vorhergesagten Spur ist wichtiger als eine
                # geringfuegig groessere Konturflaeche.
                distance_px, chosen = min(
                    nearby,
                    key=lambda entry: (
                        entry[0]
                        + 10.0 * abs(entry[1]["radius"] - (self._track_radius or entry[1]["radius"])),
                        -entry[1]["area"],
                    ),
                )
                track_valid = temporal_valid
                track_reason = "tracked" if temporal_valid else "invalid_time"
            else:
                # Die Position bleibt fuer die allgemeine Analyse erhalten,
                # wird aber nicht als Geschwindigkeitsbeobachtung akzeptiert.
                chosen = fallback
                distance_px = min(distances, key=lambda entry: entry[0])[0]
                track_valid = False
                track_reason = "contour_jump"

        center_x, center_y = chosen["center"]
        radius = chosen["radius"]
        frame.ball = (
            int(round(center_x)),
            int(round(center_y)),
            int(round(radius)),
        )
        frame.ball_track_valid = bool(track_valid)
        frame.ball_track_reason = track_reason
        frame.ball_track_distance_px = distance_px

        # Die gewaehlte Position wird immer als neue Spurposition gesetzt,
        # damit die normale Positionsanzeige nicht von einem Ausreisser
        # dauerhaft blockiert wird. Nach einem Sprung wird die
        # Geschwindigkeit aber bewusst neu angesetzt.
        previous_xy = self._track_xy
        previous_timestamp = self._track_timestamp
        self._track_xy = (center_x, center_y)
        self._track_radius = radius
        self._track_timestamp = timestamp
        self._track_misses = 0

        if (
            track_valid
            and previous_xy is not None
            and previous_timestamp is not None
        ):
            dt = timestamp - previous_timestamp
            if self.min_track_dt <= dt <= self.max_track_dt:
                instant_vx = (center_x - previous_xy[0]) / dt
                instant_vy = (center_y - previous_xy[1]) / dt
                self._track_velocity_px_s = (
                    0.5 * self._track_velocity_px_s[0] + 0.5 * instant_vx,
                    0.5 * self._track_velocity_px_s[1] + 0.5 * instant_vy,
                )
        elif not track_valid:
            self._track_velocity_px_s = (0.0, 0.0)

        return True

    def _reset_track(self) -> None:
        self._track_xy = None
        self._track_radius = None
        self._track_timestamp = None
        self._track_velocity_px_s = (0.0, 0.0)
        self._track_misses = 0




