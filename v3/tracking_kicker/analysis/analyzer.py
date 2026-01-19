import cv2
import math
import numpy as np
from typing import Any, Dict, Optional, Tuple


class KickerAnalyzer:
    def __init__(self, fps: float, goals_cfg: Dict[str, Any]):
        self.fps = float(fps)
        self.goals_cfg = goals_cfg

        # Motion
        self.prev_ball: Optional[Tuple[int, int]] = None
        self.vx_s: float = 0.0
        self.vy_s: float = 0.0
        self.speed_s: float = 0.0

        # Score
        self.score_left = 0
        self.score_right = 0

        self._frame_idx = 0
        self._cooldown_frames = max(1, int(round(0.5 * self.fps)))  # 0.5s cooldown
        self._last_goal_frame_left = -10**9
        self._last_goal_frame_right = -10**9
        self._in_goal_left = False
        self._in_goal_right = False

        self._accL = 0
        self._accR = 0
        self._acc_decay = 0.6  # wie schnell der Accumulator wieder abfällt

        # Goal mask thresholds (tuning)
        self.goal_min_px = 60          # mind. so viele Masken-Pixel im Tor-ROI
        self.goal_min_ratio = 0.0015    # oder mind. dieser Anteil vom ROI

    def _update_motion(self, x: int, y: int, predicted: bool) -> None:
        """Update speed + direction. Bei predicted nicht updaten (sonst driftet)."""
        if self.prev_ball is not None and not predicted:
            dx = x - self.prev_ball[0]
            dy = y - self.prev_ball[1]
            inst_vx = dx * self.fps
            inst_vy = dy * self.fps
            inst_speed = math.hypot(inst_vx, inst_vy)

            # smoothing
            a = 0.2
            self.vx_s = (1 - a) * self.vx_s + a * inst_vx
            self.vy_s = (1 - a) * self.vy_s + a * inst_vy
            self.speed_s = (1 - a) * self.speed_s + a * inst_speed

        self.prev_ball = (x, y)

    def _goal_hit(self, ball_mask: np.ndarray, side: str) -> bool:
        """Prüft ob im Tor-ROI genug Ball-Farb-Pixel sind."""
        h, w = ball_mask.shape[:2]
        g = self.goals_cfg[side]

        x0 = int(g["x0"] * w)
        y0 = int(g["y0"] * h)
        x1 = int(g["x1"] * w)
        y1 = int(g["y1"] * h)

        x0, x1 = sorted((max(0, x0), min(w, x1)))
        y0, y1 = sorted((max(0, y0), min(h, y1)))

        if x1 <= x0 or y1 <= y0:
            return False

        roi = ball_mask[y0:y1, x0:x1]
        cnt = int(cv2.countNonZero(roi))
        area = int(roi.size)

        thr = max(self.goal_min_px, int(self.goal_min_ratio * area))
        return cnt >= thr

    def _update_goals(self, ball_mask: Optional[np.ndarray]) -> Tuple[bool, bool]:
        """Returns (goal_left_event, goal_right_event) with cooldown + edge trigger."""
        self._frame_idx += 1
        if ball_mask is None:
            return False, False

        hitL = self._goal_hit(ball_mask, "left")
        hitR = self._goal_hit(ball_mask, "right")

        # Accumulator: kurze Treffer (1 Frame) werden addiert, dann reicht es trotzdem
        self._accL = int(self._accL * self._acc_decay) + (1 if hitL else 0)
        self._accR = int(self._accR * self._acc_decay) + (1 if hitR else 0)

        hitL_acc = self._accL >= 2  # 2 von ~3 Frames reichen
        hitR_acc = self._accR >= 2

        goalL = False
        goalR = False

        # Edge-trigger + cooldown
        if hitL and not self._in_goal_left and (self._frame_idx - self._last_goal_frame_left > self._cooldown_frames):
            self.score_left += 1
            self._last_goal_frame_left = self._frame_idx
            goalL = True
        if hitR and not self._in_goal_right and (self._frame_idx - self._last_goal_frame_right > self._cooldown_frames):
            self.score_right += 1
            self._last_goal_frame_right = self._frame_idx
            goalR = True

        self._in_goal_left = hitL
        self._in_goal_right = hitR

        return goalL, goalR

    def _draw_goal_rois(self, frame: np.ndarray) -> None:
        """Optional: Tor-ROIs einzeichnen (Debug)."""
        h, w = frame.shape[:2]
        for side, color in [("left", (0, 0, 255)), ("right", (255, 0, 0))]:
            g = self.goals_cfg[side]
            x0 = int(g["x0"] * w); y0 = int(g["y0"] * h)
            x1 = int(g["x1"] * w); y1 = int(g["y1"] * h)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)

    def _draw_arrow(self, frame: np.ndarray, x: int, y: int) -> None:
        """Pfeil für Richtung + Geschwindigkeit (aus smoothed vx/vy)."""
        speed = math.hypot(self.vx_s, self.vy_s)
        if speed < 1e-3:
            return

        # Richtung normieren
        ux = self.vx_s / speed
        uy = self.vy_s / speed

        # Pfeillänge proportional zur Geschwindigkeit (geclamped)
        arrow_len = int(max(20, min(160, speed * 0.03)))  # tuning
        x2 = int(round(x + ux * arrow_len))
        y2 = int(round(y + uy * arrow_len))

        cv2.arrowedLine(frame, (x, y), (x2, y2), (0, 255, 0), 3, tipLength=0.3)

    def draw_overlay(
        self,
        frame: np.ndarray,
        ball: Optional[Tuple[int, int, int]],
        predicted: bool = False,
        ball_mask: Optional[np.ndarray] = None,
        draw_goal_boxes: bool = False,
    ) -> None:
        h, w = frame.shape[:2]

        if draw_goal_boxes:
            self._draw_goal_rois(frame)

        # Goal check via mask + score update
        goalL_event, goalR_event = self._update_goals(ball_mask)

        # Ball + motion
        if ball:
            x, y, r = ball

            self._update_motion(x, y, predicted=predicted)

            # Ballkreis
            thickness = 2 if predicted else 3
            cv2.circle(frame, (x, y), max(2, int(r)), (0, 255, 255), thickness)

            # Pfeil
            self._draw_arrow(frame, x, y)

        # Scoreboard oben
        cv2.putText(
            frame,
            f"GOALS  LEFT: {self.score_left}   RIGHT: {self.score_right}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )

        # Speed Text
        cv2.putText(
            frame,
            f"Speed: {self.speed_s:.1f}px/s",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # Event-Text kurz einblenden
        if goalL_event:
            cv2.putText(frame, "GOAL LEFT", (w // 2 - 130, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 4)
        if goalR_event:
            cv2.putText(frame, "GOAL RIGHT", (w // 2 - 150, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 4)
