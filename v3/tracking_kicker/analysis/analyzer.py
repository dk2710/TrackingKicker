import cv2
import math
import numpy as np
import time as time
from typing import Any, Dict, Optional, Tuple


class KickerAnalyzer:
    def __init__(self, fps: float, goals_cfg: Dict[str, Any], field_cfg: Optional[Dict[str, Any]] = None):

        self.fps = float(fps)
        self.goals_cfg = goals_cfg

        # Ball motion (nur aus Messungen!)
        self.prev_xy: Optional[Tuple[int, int]] = None
        self.vx_s: float = 0.0  # px/s
        self.vy_s: float = 0.0  # px/s
        self.speed_s: float = 0.0

        self._speed_peak = 0.0
        self._speed_display = 0
        self._speed_window_start = time.perf_counter()
        self._speed_window = 0.5

        # Score
        self.score_left = 0
        self.score_right = 0

        # Goal detection tuning
        self.goal_min_px = 60        # runter = sensibler
        self.goal_min_ratio = 0.0015 # runter = sensibler

        # Robustheit gegen "nur 1 Frame im Tor"
        self._accL = 0
        self._accR = 0
        self._acc_decay = 0.6   # 0..1, höher = länger "merken"
        self._frame_idx = 0
        self._cooldown_frames = max(1, int(round(1 * self.fps)))  # 0.5s cooldown
        self._last_goal_L = -10**9
        self._last_goal_R = -10**9
        self._in_goal_L = False
        self._in_goal_R = False

        # --- Goal confirmation: erst "pending", dann bestätigen wenn Ball 1s weg ist ---
        self.goal_confirm_missing_frames = max(1, int(round(1.0 * self.fps)))  # 1 Sekunde
        self.pending_goal_side: Optional[str] = None   # "left" / "right" / None
        self.pending_miss = 0                          # wie viele Frames Ball nicht erkannt wurde
        self.pending_started_frame = -10**9            # Frame-Index, wann pending gestartet wurde

        # optional: cooldown lieber auch ~1s, damit nichts doppelt zählt
        self._cooldown_frames = max(self._cooldown_frames, self.goal_confirm_missing_frames)


        # Debug: cnt/thr pro Seite anzeigen
        self.last_cntL = 0
        self.last_thrL = 0
        self.last_cntR = 0
        self.last_thrR = 0

        # Debug Anzeige "festhalten" (damit man schnelle Tore ablesen kann)
        self.debug_hold_frames = int(round(2.0 * self.fps))  # 2 Sekunden halten
        self._debug_hold = 0

        # Peak innerhalb der Hold-Phase (hilft bei 1-Frame-Treffern)
        self.peak_cntL = 0
        self.peak_cntR = 0


        # Pixel-Accumulator statt 0/1 Hits (hilft bei schnellen Toren)
        self._accL_px = 0.0
        self._accR_px = 0.0


        self.field_cfg = field_cfg
        self.H_img2cm = None
        self.prev_world = None
        self.speed_kmh_s = 0.0

    def _update_motion(self, x: int, y: int) -> None:
        if self.prev_xy is not None:
            dx = x - self.prev_xy[0]
            dy = y - self.prev_xy[1]

            inst_vx = dx * self.fps
            inst_vy = dy * self.fps

            inst_speed = math.hypot(inst_vx, inst_vy)

            a = 0.2
            self.vx_s = (1 - a) * self.vx_s + a * inst_vx
            self.vy_s = (1 - a) * self.vy_s + a * inst_vy
            self.speed_s = (1 - a) * self.speed_s + a * inst_speed

        self.prev_xy = (x, y)

    def _goal_cnt_thr(self, ball_mask: np.ndarray, side: str) -> Tuple[int, int]:
        h, w = ball_mask.shape[:2]
        g = self.goals_cfg[side]

        x0 = int(g["x0"] * w);
        y0 = int(g["y0"] * h)
        x1 = int(g["x1"] * w);
        y1 = int(g["y1"] * h)

        x0, x1 = sorted((max(0, x0), min(w, x1)))
        y0, y1 = sorted((max(0, y0), min(h, y1)))

        if x1 <= x0 or y1 <= y0:
            return 0, 10 ** 9

        roi = ball_mask[y0:y1, x0:x1]
        cnt = int(cv2.countNonZero(roi))
        area = int(roi.size)

        thr = max(self.goal_min_px, int(self.goal_min_ratio * area))
        return cnt, thr

    def _update_goals(self, ball_mask: Optional[np.ndarray], ball: Optional[Tuple[int, int, int]], frame_shape) -> \
    Tuple[bool, bool]:
        self._frame_idx += 1

        if self._debug_hold > 0:
            self._debug_hold -= 1
        else:
            # wenn Hold abgelaufen: Peaks zurücksetzen
            self.peak_cntL = 0
            self.peak_cntR = 0


        # 1) hits aus Maske (cnt/thr) + Debug setzen
        hitL = False
        hitR = False

        if ball_mask is not None:
            cntL, thrL = self._goal_cnt_thr(ball_mask, "left")
            cntR, thrR = self._goal_cnt_thr(ball_mask, "right")

            # Debug speichern + Hold zurücksetzen
            self.last_cntL, self.last_thrL = cntL, thrL
            self.last_cntR, self.last_thrR = cntR, thrR
            self._debug_hold = self.debug_hold_frames

            # Peak merken (max während Hold-Zeit)
            self.peak_cntL = max(self.peak_cntL, cntL)
            self.peak_cntR = max(self.peak_cntR, cntR)

            # optional: hilft bei schnellen/blur Schüssen (Maske im Tor leicht "dicker")
            # Wenn du das willst, müssen wir cnt/thr auf dilated ROI berechnen.
            # Minimal-Variante: Schwellen über cnt/thr lassen:
            hitL = (cntL >= thrL)
            hitR = (cntR >= thrR)
        else:
            self.last_cntL = self.last_thrL = 0
            self.last_cntR = self.last_thrR = 0

        # 2) Falls noch kein pending aktiv ist: pending starten, wenn Hit erkannt wurde
        if self.pending_goal_side is None:
            # Cooldown check
            canL = (self._frame_idx - self._last_goal_L > self._cooldown_frames)
            canR = (self._frame_idx - self._last_goal_R > self._cooldown_frames)

            # Wenn beide true sind: über Ballposition entscheiden, sonst Priorität
            if hitL and canL and (not hitR or not canR):
                self.pending_goal_side = "left"
                self.pending_miss = 0
                self.pending_started_frame = self._frame_idx
                print(
                    f"[{self._frame_idx}] pending LEFT  cnt/thr={self.last_cntL}/{self.last_thrL}  peak={self.peak_cntL}")

            elif hitR and canR and (not hitL or not canL):
                self.pending_goal_side = "right"
                self.pending_miss = 0
                self.pending_started_frame = self._frame_idx
                print(f"[{self._frame_idx}] pending RIGHT cnt/thr={self.last_cntR}/{self.last_thrR} peak={self.peak_cntR}")

            elif hitL and hitR and canL and canR:
                # beide gleichzeitig: wenn Ball vorhanden, nimm das Goal, in dessen ROI der Ballmittelpunkt liegt
                if ball is not None:
                    x, y, _ = ball
                    if self._point_in_goal_roi(x, y, "left", frame_shape):
                        self.pending_goal_side = "left"
                    elif self._point_in_goal_roi(x, y, "right", frame_shape):
                        self.pending_goal_side = "right"
                    else:
                        # fallback: links bevorzugen
                        self.pending_goal_side = "left"
                else:
                    self.pending_goal_side = "left"
                self.pending_miss = 0
                self.pending_started_frame = self._frame_idx

        # 3) Pending-Phase: bestätigen, wenn Ball 1s nicht gefunden wird
        goalL = False
        goalR = False

        if self.pending_goal_side is not None:
            if ball is None:
                self.pending_miss += 1
            else:
                # Ball wieder da: wenn eindeutig NICHT im Goal-ROI, pending abbrechen (war vermutlich false positive)
                x, y, _ = ball
                inL = self._point_in_goal_roi(x, y, "left", frame_shape)
                inR = self._point_in_goal_roi(x, y, "right", frame_shape)

                if not inL and not inR:
                    # Ball ist wieder im Spielfeld -> pending cancel
                    self.pending_goal_side = None
                    self.pending_miss = 0
                else:
                    # Ball ist weiterhin im Goal-Bereich sichtbar -> pending bleibt, miss bleibt 0
                    self.pending_miss = 0

            # Bestätigung: Ball 1 Sekunde nicht gesehen
            if self.pending_goal_side is not None and self.pending_miss >= self.goal_confirm_missing_frames:
                if self.pending_goal_side == "left":
                    self.score_left += 1
                    self._last_goal_L = self._frame_idx
                    goalL = True
                else:
                    self.score_right += 1
                    self._last_goal_R = self._frame_idx
                    goalR = True

                # pending reset
                self.pending_goal_side = None
                self.pending_miss = 0



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

    def draw_overlay(self, frame, ball, ball_mask=None, draw_goal_boxes=False):
        h, w = frame.shape[:2]

        # Tor-ROIs optional zeichnen
        if draw_goal_boxes:
            for side, color in [("left", (0, 0, 255)), ("right", (255, 0, 0))]:
                g = self.goals_cfg[side]
                x0 = int(g["x0"] * w);
                y0 = int(g["y0"] * h)
                x1 = int(g["x1"] * w);
                y1 = int(g["y1"] * h)
                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 4)

        # Torerkennung über Maske
        goalL, goalR = self._update_goals(ball_mask, ball, frame.shape)

        cv2.putText(frame, f"GoalMask L: {self.last_cntL}/{self.last_thrL}  peak:{self.peak_cntL}",
                    (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"GoalMask R: {self.last_cntR}/{self.last_thrR}  peak:{self.peak_cntR}",
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Ball zeichnen + Bewegung nur wenn Ball vorhanden
        if ball is not None:
            x, y, r = ball

            # Homography (für km/h) vorbereiten
            self._ensure_homography(frame.shape)

            # Pixel-Speed/Richtung
            self._update_motion(x, y)

            # km/h-Speed (wenn field_cfg vorhanden)
            world = self._img_to_cm(x, y)
            if world is not None:
                if self.prev_world is not None:
                    dX = world[0] - self.prev_world[0]  # cm
                    dY = world[1] - self.prev_world[1]  # cm
                    inst_cm_s = math.hypot(dX, dY) * self.fps
                    inst_kmh = inst_cm_s * 0.036

                    now = time.perf_counter()

                    if inst_kmh > self._speed_peak:
                        self._speed_peak = inst_kmh

                    if now - self._speed_window_start >= self._speed_window:
                        self._speed_display = int(round(self._speed_peak))
                        self._speed_peak = 0.0
                        self._speed_window_start = now

                    a = 0.2
                    self.speed_kmh_s = (1 - a) * self.speed_kmh_s + a * inst_kmh
                self.prev_world = world

            cv2.circle(frame, (x, y), max(2, int(r)), (0, 255, 255), 3)
            self._draw_arrow(frame, x, y)

        else:
            # wichtig: verhindert "Speed-Spikes", wenn Ball mal fehlt
            self.prev_xy = None
            self.prev_world = None

        # UI Texte (Score, Speed etc.)
        cv2.putText(frame, f"GOALS  LEFT: {self.score_left}   RIGHT: {self.score_right}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.putText(frame, f"Speed: {self._speed_display} km/h",
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if goalL:
            cv2.putText(frame, "GOAL LEFT", (w // 2 - 130, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 4)
        if goalR:
            cv2.putText(frame, "GOAL RIGHT", (w // 2 - 150, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 4)

    def _ensure_homography(self, frame_shape) -> None:
        if self.field_cfg is None or self.H_img2cm is not None:
            return

        h, w = frame_shape[:2]
        corners_norm = self.field_cfg["corners_norm"]  # [OL, OR, UR, UL]
        fw = float(self.field_cfg["size_cm"]["w"])
        fh = float(self.field_cfg["size_cm"]["h"])

        src = np.array([[cn[0] * w, cn[1] * h] for cn in corners_norm], dtype=np.float32)
        dst = np.array([[0.0, 0.0], [fw, 0.0], [fw, fh], [0.0, fh]], dtype=np.float32)

        self.H_img2cm = cv2.getPerspectiveTransform(src, dst)

    def _img_to_cm(self, x: int, y: int) -> Optional[Tuple[float, float]]:
        if self.H_img2cm is None:
            return None
        pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.H_img2cm)[0][0]
        return float(out[0]), float(out[1])

    def _point_in_goal_roi(self, x: int, y: int, side: str, frame_shape) -> bool:
        h, w = frame_shape[:2]
        g = self.goals_cfg[side]

        x0 = int(g["x0"] * w); y0 = int(g["y0"] * h)
        x1 = int(g["x1"] * w); y1 = int(g["y1"] * h)

        xmin, xmax = sorted((x0, x1))
        ymin, ymax = sorted((y0, y1))
        return (xmin <= x <= xmax) and (ymin <= y <= ymax)
