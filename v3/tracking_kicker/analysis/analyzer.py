from xml.etree.ElementTree import tostring

import cv2
import math
import numpy as np
import time as time
from typing import Any, Dict, Optional, Tuple

from v3.tracking_kicker.model.Vision import FieldConfig, GoalsConfig
from v3.tracking_kicker.analysis.rod_control import RodControlTracker



class KickerAnalyzer:
    def __init__(self, goals_cfg: GoalsConfig, field_cfg: FieldConfig | None = None):

        self.prev_ball_frame = None
        self.goals_cfg = goals_cfg

        # Ball motion (nur aus Messungen!)
        self.prev_xy: Optional[Tuple[int, int]] = None
        self.vx_s: float = 0.0  # px/s
        self.vy_s: float = 0.0  # px/s
        self.speed_s: float = 0.0

        self._speed_display = 0
        self._speed_window_start = time.perf_counter()
        self._speed_window = 0.1
        self.min_dt = 0.015  # 15 ms
        self.prev_ball_time = None

        self.ball_history = []  # Liste aus (timestamp, world_x, world_y)
        self.history_size = 8  # über 8 Frames mitteln

        # Geschwindigkeitsfenster: ausschliesslich die letzte Sekunde.
        self.speed_analysis_seconds = 1.0
        self.world_buffer = []
        self.time_buffer = []

        self._speed_peak = 0.0
        self._speed_peak_time = time.perf_counter()
        self._speed_peak_hold = 1 # 0.5 Sekunden halten

        # Score
        self.score_left = 0
        self.score_right = 0


        self._pending_side = None  # "left" | "right" | None
        self._pending_start_time = 0.0
        self._pending_seen_ball = False
        self._goal_locked = False

        #Zeitmessung
        self._last_t: float | None = None


        #Bereicherkennung
        self.field_cfg = field_cfg
        self.rod_control = None
        self.H_img2cm = None
        self.prev_world = None
        self.speed_kmh_s = 0.0

        if self.field_cfg is not None and getattr(self.field_cfg, "rods", None) is not None:
            self.rod_control = RodControlTracker(
                rods_x_norm=self.field_cfg.rods.xs_norm
            )


    def _update_goals(self, ball, frame_shape):
        goalL = goalR = False

        if ball is None:
            # Ball weg → mögliches Tor bestätigen
            if self._pending_side is not None and not self._goal_locked:
                if self._pending_side == "left":
                    self.score_left += 1
                    goalL = True
                else:
                    self.score_right += 1
                    goalR = True

                self._goal_locked = True
                self._pending_side = None

            return goalL, goalR

        x, y, r = ball

        inL = self._ball_in_goal_roi(x, y, r, "left", frame_shape)
        inR = self._ball_in_goal_roi(x, y, r, "right", frame_shape)

        # LOCKED: warten bis Ball wieder im Feld ist
        if self._goal_locked:
            if not inL and not inR:
                self._goal_locked = False
            return False, False

        # Ball betritt Tor
        if self._pending_side is None:
            if inL:
                self._pending_side = "left"
            elif inR:
                self._pending_side = "right"
            return False, False

        # Ball war im Tor, ist jetzt draußen → TOR
        if self._pending_side == "left" and not inL:
            self.score_left += 1
            goalL = True
            self._goal_locked = True
            self._pending_side = None

        elif self._pending_side == "right" and not inR:
            self.score_right += 1
            goalR = True
            self._goal_locked = True
            self._pending_side = None

        return goalL, goalR

    def _draw_arrow(self, frame: np.ndarray, x: int, y: int) -> None:
        speed = math.hypot(self.vx_s, self.vy_s)
        if speed < 1e-3:
            return

        ux = self.vx_s / speed
        uy = self.vy_s / speed

        arrow_len = int(max(20, min(160, speed * 0.03)))  # anpassen falls nötig
        x2 = int(round(x + ux * arrow_len))
        y2 = int(round(y + uy * arrow_len))

        cv2.arrowedLine(frame, (x, y), (x2, y2), (0, 255, 0), 3, tipLength=0.3)

    def process_frame(self, frame):
        h, w = frame.frame.shape[:2]

        # Torerkennung
        goalL, goalR = self._update_goals(frame.ball, frame.frame.shape)

        # Rod-Control
        if self.rod_control is not None and frame.ball is not None:
            x, y, r = frame.ball
            x_norm = x / w
            self.rod_control.update(x_norm)

        # Homography vorbereiten
        self._ensure_homography(frame.frame.shape)


        # Das Geschwindigkeitsfenster wird zeitbasiert unten begrenzt.

        # Ball erkannt?
        if frame.ball is not None:
            x, y, r = frame.ball
            world = self._img_to_cm(x, y)
            self.world_buffer.append(world)
        else:
            # Ball nicht erkannt → None eintragen
            self.world_buffer.append(None)

        # Zeitstempel immer speichern
        self.time_buffer.append(frame.timestamp)

        # Buffer auf die letzte Sekunde begrenzen
        while (
                len(self.time_buffer) > 2
                and self.time_buffer[-1] - self.time_buffer[0]
                > self.speed_analysis_seconds
        ):
            self.world_buffer.pop(0)
            self.time_buffer.pop(0)

        # Messen, sobald mindestens zwei Punkte im Ein-Sekunden-Fenster liegen
        if len(self.world_buffer) >= 2:

            # Alle echten Ballpunkte extrahieren
            valid_points = [(i, p) for i, p in enumerate(self.world_buffer) if p is not None]

            # Wenn weniger als 2 echte Punkte → keine Messung möglich
            if len(valid_points) < 2:
                self.speed_kmh_s = 0.0
                return goalL, goalR

            # Start = erster echter Punkt
            start_index, start = valid_points[0]

            # End = letzter echter Punkt
            end_index, end = valid_points[-1]

            # Distanz
            dX = end[0] - start[0]
            dY = end[1] - start[1]
            dist_cm = math.hypot(dX, dY)

            # Zeit zwischen diesen beiden Frames
            dt = self.time_buffer[end_index] - self.time_buffer[start_index]

            if dt >= self.min_dt:
                inst_cm_s = dist_cm / dt
                inst_kmh = inst_cm_s * 0.036

                # ECHTEN Wert speichern
                self.speed_kmh_s = inst_kmh

                # --- PEAK HOLD LOGIK ---
                now = time.perf_counter()

                # Peak aktualisieren, wenn neuer Wert größer ist
                if inst_kmh > self._speed_peak:
                    self._speed_peak = inst_kmh
                    self._speed_peak_time = now

                    print("X 1.", end[0], "X 3.", start[0], )
                    print("Y 1.", end[1], "Y 3.", start[1], )
                    print("dist_cm:", dist_cm)
                    print("dt:", self.time_buffer[end_index], self.time_buffer[start_index])
                    print("inst_kmh:", inst_kmh)

                # Peak zurücksetzen, wenn 0.5s vergangen sind
                if now - self._speed_peak_time > self._speed_peak_hold:
                    self._speed_peak = inst_kmh

                # Anzeige = Peak (optional geglättet)
                a = 0.2
                self._speed_display = (1 - a) * self._speed_display + a * self._speed_peak


        return goalL, goalR

    def draw_overlay(self, frame, draw_goal_boxes=False):
        h, w = frame.frame.shape[:2]

        # Tor-ROIs optional zeichnen
        if draw_goal_boxes:
            for side, color in [("left", (0, 0, 255)), ("right", (255, 0, 0))]:
                g = self.goals_cfg.left if side == "left" else self.goals_cfg.right
                x0 = int(g.x0 * w)
                y0 = int(g.y0 * h)
                x1 = int(g.x1 * w)
                y1 = int(g.y1 * h)
                cv2.rectangle(frame.frame, (x0, y0), (x1, y1), color, 4)

        # Ball zeichnen
        if frame.ball is not None:
            x, y, r = frame.ball
            cv2.circle(frame.frame, (int(x), int(y)), max(2, int(r)), (0, 255, 255), 3)
            self._draw_arrow(frame.frame, int(x), int(y))

        # Rod-Control anzeigen
        if self.rod_control is not None:
            for i, cnt in enumerate(self.rod_control.get_counts()):
                cv2.putText(
                    frame.frame,
                    f"Rod {i + 1}: {cnt}",
                    (20, 100 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

        # Score
        cv2.putText(frame.frame, f"GOALS LEFT: {self.score_left} RIGHT: {self.score_right}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Speed
        cv2.putText(frame.frame, f"Speed: {self._speed_display} km/h",
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def _ensure_homography(self, frame_shape) -> None:
        if self.field_cfg is None or self.H_img2cm is not None:
            return

        h, w = frame_shape[:2]
        corners_norm = self.field_cfg.corners_norm  # [OL, OR, UR, UL]
        fw = float(self.field_cfg.size_cm.w)
        fh = float(self.field_cfg.size_cm.h)

        src = np.array([[cn[0] * w, cn[1] * h] for cn in corners_norm], dtype=np.float32)
        dst = np.array([[0.0, 0.0], [fw, 0.0], [fw, fh], [0.0, fh]], dtype=np.float32)

        self.H_img2cm = cv2.getPerspectiveTransform(src, dst)

    def _img_to_cm(self, x: int, y: int) -> Optional[Tuple[float, float]]:
        if self.H_img2cm is None:
            return None
        pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
        #SPEICHERUNG DER X UND Y WERTE FÜR POSITION
        out = cv2.perspectiveTransform(pt, self.H_img2cm)[0][0]
        return float(out[0]), float(out[1])

    def _ball_in_goal_roi(self, x, y, r, side, frame_shape) -> bool:
        h, w = frame_shape[:2]
        g = self.goals_cfg.left if side == "left" else self.goals_cfg.right

        x0 = int(g.x0 * w);
        y0 = int(g.y0 * h)
        x1 = int(g.x1 * w);
        y1 = int(g.y1 * h)

        xmin, xmax = sorted((x0, x1))
        ymin, ymax = sorted((y0, y1))

        return (
                (x + r >= xmin) and (x - r <= xmax) and
                (y + r >= ymin) and (y - r <= ymax)
        )

    def _ball_zone(self, x: int, frame_shape) -> int | None:
        if self.field_cfg is None or self.field_cfg.rods is None:
            return None

        _, w = frame_shape[:2]
        xn = x / w

        rods = sorted(self.field_cfg.rods.xs_norm)

        for i, rx in enumerate(rods):
            if xn < rx:
                return i

        return len(rods)  # letzte Zone

    @property
    def speed_display(self):
        return self._speed_display
