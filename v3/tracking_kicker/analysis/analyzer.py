import cv2
import math
import time
from typing import Optional, Tuple

import numpy as np

from v3.tracking_kicker.analysis.rod_control import RodControlTracker
from v3.tracking_kicker.model.Vision import FieldConfig, GoalsConfig


GOAL_CONFIRMATION_MISSING_SECONDS = 1.0


class KickerAnalyzer:
    """Stateful goal detection plus the retained legacy analysis helpers.

    The active application calls :meth:`process_goal_frame` during the live
    stream.  The expensive position and speed path remains available for
    replay tools, while the goal media worker performs the current reverse
    three-point speed calculation.
    """

    def __init__(self, goals_cfg: GoalsConfig, field_cfg: FieldConfig | None = None):
        self.goals_cfg = goals_cfg
        self.field_cfg = field_cfg

        # Ball motion used by the live overlay.
        self.prev_xy: Optional[Tuple[int, int]] = None
        self.vx_s = 0.0
        self.vy_s = 0.0
        self.speed_s = 0.0

        self._speed_display = 0.0
        self.speed_analysis_seconds = 1.0
        # Keep a little more than the analysis window. Goal confirmation can
        # happen after a short run of missing frames.
        self.speed_buffer_seconds = 2.0
        # Two frames at 180 FPS are about 5.6 ms apart. Values below 2.5 ms
        # are treated as duplicate/bad timestamps, while a gap above 250 ms
        # is too large for a reliable instantaneous speed segment.
        self.min_dt = 0.0025
        self.max_dt = 0.25
        # Deliberately generous for a table-football shot. This rejects
        # contour jumps while retaining very fast real shots.
        self.max_plausible_speed_kmh = 350.0

        # Public buffers are kept for compatibility with the existing code.
        self.world_buffer = []
        self.time_buffer = []
        # Only detector-validated observations are stored here. Raw
        # frame.ball remains available to the general position analysis.
        self.ball_observations = []  # (timestamp, world_x, world_y)
        self.last_speed_pair_count = 0
        self.last_rejected_speed_pair_count = 0

        self._speed_peak = 0.0
        self._speed_peak_time = time.perf_counter()
        self._speed_peak_hold = 1.0

        # Score and goal state.
        self.score_left = 0
        self.score_right = 0
        self._pending_side = None  # "left" | "right" | None
        self._pending_start_time: Optional[float] = None
        self._pending_missing_since: Optional[float] = None
        self._goal_locked = False
        self._candidate_event_id = 0
        self._pending_candidate = None
        self._cancelled_candidate = None
        self._last_reported_candidate_id = None

        # The application uses this event as the payload for its POST.
        self._goal_event_id = 0
        self.last_goal_event = None
        self._last_goal_timestamp: Optional[float] = None

        self.rod_control = None
        self.H_img2cm = None
        self.speed_kmh_s = 0.0

        if self.field_cfg is not None and getattr(self.field_cfg, "rods", None) is not None:
            self.rod_control = RodControlTracker(
                rods_x_norm=self.field_cfg.rods.xs_norm
            )

    def _emit_goal(self, side: str, timestamp: Optional[float]) -> None:
        """Create one goal event at the first frame seen inside the goal."""
        if side == "left":
            self.score_left += 1
        else:
            self.score_right += 1

        goal_time = float(timestamp) if timestamp is not None else time.perf_counter()
        self._goal_event_id += 1
        self._last_goal_timestamp = goal_time
        candidate_id = (
            self._pending_candidate.get("candidate_id")
            if self._pending_candidate is not None
            else None
        )
        self.last_goal_event = {
            "event_id": self._goal_event_id,
            "event": "goal",
            "side": side,
            "goal_side": side,
            "goal": side,
            "position": side,
            "goal_position": side,
            "goal_timestamp": goal_time,
            "candidate_id": candidate_id,
            "speed_kmh": 0.0,
            "ball_speed_kmh": 0.0,
            "speed": 0.0,
            "score_left": self.score_left,
            "score_right": self.score_right,
        }

    def _update_goals(
        self,
        ball,
        frame_shape,
        timestamp: Optional[float] = None,
    ):
        goal_left = goal_right = False
        timestamp = float(timestamp) if timestamp is not None else time.perf_counter()

        if ball is None:
            # A goal is only confirmed after a complete second without any
            # detected ball. This avoids counting a short detector gap as a
            # goal immediately after the ball enters the goal ROI.
            if self._pending_side is not None and not self._goal_locked:
                if self._pending_missing_since is None:
                    self._pending_missing_since = timestamp
                elif (
                    timestamp - self._pending_missing_since
                    >= GOAL_CONFIRMATION_MISSING_SECONDS
                ):
                    side = self._pending_side
                    self._emit_goal(
                        side,
                        self._pending_start_time
                        if self._pending_start_time is not None
                        else timestamp,
                    )
                    goal_left = side == "left"
                    goal_right = side == "right"
                    self._goal_locked = True
                    self._pending_side = None
                    self._pending_start_time = None
                    self._pending_missing_since = None
                    self._pending_candidate = None
            return goal_left, goal_right

        x, y, r = ball
        in_left = self._ball_in_goal_roi(x, y, r, "left", frame_shape)
        in_right = self._ball_in_goal_roi(x, y, r, "right", frame_shape)

        # Wait until the ball has left both goal ROIs before accepting another
        # goal. This prevents several notifications for one shot.
        if self._goal_locked:
            if not in_left and not in_right:
                self._goal_locked = False
            return False, False

        if self._pending_side is None:
            if in_left:
                self._pending_side = "left"
                self._pending_start_time = timestamp
                self._pending_missing_since = None
                self._candidate_event_id += 1
                self._pending_candidate = {
                    "candidate_id": self._candidate_event_id,
                    "side": "left",
                    "goal_timestamp": timestamp,
                }
            elif in_right:
                self._pending_side = "right"
                self._pending_start_time = timestamp
                self._pending_missing_since = None
                self._candidate_event_id += 1
                self._pending_candidate = {
                    "candidate_id": self._candidate_event_id,
                    "side": "right",
                    "goal_timestamp": timestamp,
                }
            return False, False

        pending_in_goal = (
            in_left if self._pending_side == "left" else in_right
        )
        if pending_in_goal:
            # The ball is still visible, so the one-second absence window has
            # not started (or was interrupted).
            self._pending_missing_since = None
            return False, False

        # A visible ball outside the candidate ROI means that the required
        # one-second complete absence has not happened. Cancel this candidate;
        # if it is already in the other ROI, start that side as a new one.
        if self._pending_candidate is not None:
            self._cancelled_candidate = self._pending_candidate
        self._pending_candidate = None
        self._pending_side = None
        self._pending_start_time = None
        self._pending_missing_since = None
        if in_left:
            self._pending_side = "left"
            self._pending_start_time = timestamp
            self._candidate_event_id += 1
            self._pending_candidate = {
                "candidate_id": self._candidate_event_id,
                "side": "left",
                "goal_timestamp": timestamp,
            }
        elif in_right:
            self._pending_side = "right"
            self._pending_start_time = timestamp
            self._candidate_event_id += 1
            self._pending_candidate = {
                "candidate_id": self._candidate_event_id,
                "side": "right",
                "goal_timestamp": timestamp,
            }

        return goal_left, goal_right

    def _draw_arrow(self, frame: np.ndarray, x: int, y: int) -> None:
        speed = math.hypot(self.vx_s, self.vy_s)
        if speed < 1e-3:
            return

        ux = self.vx_s / speed
        uy = self.vy_s / speed
        arrow_len = int(max(20, min(160, speed * 0.03)))
        x2 = int(round(x + ux * arrow_len))
        y2 = int(round(y + uy * arrow_len))
        cv2.arrowedLine(frame, (x, y), (x2, y2), (0, 255, 0), 3, tipLength=0.3)

    def _append_observation(
        self,
        timestamp: float,
        ball,
        frame_shape,
        speed_valid: bool = True,
    ) -> None:
        # A raw ball position is useful to the normal analysis, but it must
        # not enter the goal-speed history if the detector saw a contour jump
        # or an invalid frame interval.
        if ball is None or not speed_valid:
            return

        x, y, _ = ball
        world = self._img_to_cm(x, y)
        if world is None:
            return

        timestamp = float(timestamp)
        observation = (timestamp, world[0], world[1])
        self.ball_observations.append(observation)

        # Recovery frames are normally fed in order. Sorting here also makes
        # the method safe if a late frame arrives from a bounded queue.
        if len(self.ball_observations) >= 2 and (
            self.ball_observations[-2][0] > self.ball_observations[-1][0]
        ):
            self.ball_observations.sort(key=lambda item: item[0])

        cutoff = timestamp - self.speed_buffer_seconds
        self.ball_observations = [
            item for item in self.ball_observations if item[0] >= cutoff
        ]

        # Compatibility buffers contain only valid observations. The actual
        # frame timestamp still determines the speed interval below.
        self.world_buffer.append(world)
        self.time_buffer.append(timestamp)
        while self.time_buffer and self.time_buffer[0] < cutoff:
            self.time_buffer.pop(0)
            self.world_buffer.pop(0)

    def calculate_speed_kmh(self, reference_timestamp: float) -> float:
        """Return the fastest validated segment in the preceding second.

        Two valid detections are sufficient for a fast shot. Every segment is
        independently checked for a sensible timestamp interval and a
        physically plausible distance. A single invalid contour jump can
        therefore no longer create a huge speed value.
        """
        end_time = float(reference_timestamp)
        start_time = end_time - self.speed_analysis_seconds
        points = sorted([
            item for item in self.ball_observations
            if start_time <= item[0] <= end_time
        ], key=lambda item: item[0])

        speeds = []
        rejected = 0
        for first, second in zip(points, points[1:]):
            dt = second[0] - first[0]
            if not (self.min_dt <= dt <= self.max_dt):
                rejected += 1
                continue

            distance_cm = math.hypot(
                second[1] - first[1],
                second[2] - first[2],
            )
            speed = distance_cm / dt * 0.036
            if not math.isfinite(speed) or speed > self.max_plausible_speed_kmh:
                rejected += 1
                continue
            speeds.append(speed)

        self.last_speed_pair_count = len(speeds)
        self.last_rejected_speed_pair_count = rejected
        if not speeds:
            return 0.0
        return max(speeds)

    def observe_recovered_frame(self, frame) -> None:
        """Add a late-detected ball without replaying goal state out of order.

        A skipped frame can be older than the live frame that has already been
        processed. Its position is still useful for speed, but feeding it
        through the stateful goal machine again would make the state travel
        backwards in time. Goal confirmation for such a frame is therefore
        handled explicitly by :meth:`confirm_recovered_goal`.
        """
        self._ensure_homography(frame.frame.shape)
        timestamp = float(getattr(frame, "timestamp", time.perf_counter()))
        self._append_observation(
            timestamp,
            frame.ball,
            frame.frame.shape,
            speed_valid=getattr(frame, "ball_track_valid", True),
        )

    def confirm_recovered_goal(self, side: str, landing_timestamp: float):
        """Register a recovered goal candidate for normal confirmation.

        Recovered frames are older than the current live frame. They must not
        emit a goal immediately: the live state machine still has to observe
        one full second with no detected ball.
        """
        if self._goal_locked:
            return False, False
        if self._pending_side is not None and self._pending_side != side:
            return False, False

        if self._pending_side is None:
            self._pending_side = side
            self._pending_start_time = float(landing_timestamp)
            self._pending_missing_since = None
        return False, False

    def consume_goal_candidate(self):
        """Return and clear the newest candidate that entered a goal ROI.

        The worker uses this early event to freeze the three-second history
        before it rotates out.  The existing one-second confirmation rule
        still decides whether the candidate becomes a real goal.
        """

        candidate = self._pending_candidate
        if candidate is None:
            return None
        candidate_id = candidate.get("candidate_id")
        if candidate_id == self._last_reported_candidate_id:
            return None
        self._last_reported_candidate_id = candidate_id
        return dict(candidate)

    def consume_cancelled_goal_candidate(self):
        """Return a candidate that left the ROI before confirmation."""

        candidate = self._cancelled_candidate
        self._cancelled_candidate = None
        return candidate

    def process_goal_frame(self, frame):
        """Run only the lightweight live goal state machine.

        Continuous position and speed analysis is intentionally disabled for
        now.  Ball detection remains active because it is the trigger that
        freezes the history and confirms a goal; the expensive speed analysis
        is performed retrospectively by :mod:`goal_media`.
        """

        timestamp = float(getattr(frame, "timestamp", time.perf_counter()))
        return self._update_goals(frame.ball, frame.frame.shape, timestamp)

    def _finalize_goal_speed(self) -> None:
        if self._last_goal_timestamp is None:
            return
        goal_speed = self.calculate_speed_kmh(self._last_goal_timestamp)
        self.speed_kmh_s = goal_speed
        self._speed_display = goal_speed
        if self.last_goal_event is not None:
                self.last_goal_event["speed_kmh"] = float(goal_speed)
                self.last_goal_event["ball_speed_kmh"] = float(goal_speed)
                self.last_goal_event["speed"] = float(goal_speed)
                self.last_goal_event["speed_pair_count"] = self.last_speed_pair_count
                self.last_goal_event["speed_rejected_pair_count"] = self.last_rejected_speed_pair_count

    def process_frame(self, frame):
        _, w = frame.frame.shape[:2]
        timestamp = float(getattr(frame, "timestamp", time.perf_counter()))

        goal_left, goal_right = self._update_goals(
            frame.ball,
            frame.frame.shape,
            timestamp,
        )

        if self.rod_control is not None and frame.ball is not None:
            x, _, _ = frame.ball
            self.rod_control.update(x / w)

        self._ensure_homography(frame.frame.shape)
        self._append_observation(
            timestamp,
            frame.ball,
            frame.frame.shape,
            speed_valid=getattr(frame, "ball_track_valid", True),
        )

        if goal_left or goal_right:
            # The event timestamp is the first in-goal frame, not the later
            # frame in which the ball disappeared.
            self._finalize_goal_speed()

        return goal_left, goal_right

    def draw_overlay(self, frame, draw_goal_boxes=False):
        _, w = frame.frame.shape[:2]

        if draw_goal_boxes:
            for side, color in [("left", (0, 0, 255)), ("right", (255, 0, 0))]:
                goal = self.goals_cfg.left if side == "left" else self.goals_cfg.right
                x0 = int(goal.x0 * w)
                y0 = int(goal.y0 * frame.frame.shape[0])
                x1 = int(goal.x1 * w)
                y1 = int(goal.y1 * frame.frame.shape[0])
                cv2.rectangle(frame.frame, (x0, y0), (x1, y1), color, 4)

        if frame.ball is not None:
            x, y, r = frame.ball
            cv2.circle(frame.frame, (int(x), int(y)), max(2, int(r)), (0, 255, 255), 3)
            self._draw_arrow(frame.frame, int(x), int(y))

        if self.rod_control is not None:
            for i, count in enumerate(self.rod_control.get_counts()):
                cv2.putText(
                    frame.frame,
                    f"Rod {i + 1}: {count}",
                    (20, 100 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

        cv2.putText(
            frame.frame,
            f"GOALS LEFT: {self.score_left} RIGHT: {self.score_right}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame.frame,
            (
                f"Last goal: {self._speed_display:.1f} km/h"
                if self._last_goal_timestamp is not None
                else "Last goal: --"
            ),
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    def _ensure_homography(self, frame_shape) -> None:
        if self.field_cfg is None or self.H_img2cm is not None:
            return

        h, w = frame_shape[:2]
        corners_norm = self.field_cfg.corners_norm
        field_width = float(self.field_cfg.size_cm.w)
        field_height = float(self.field_cfg.size_cm.h)
        src = np.array(
            [[corner[0] * w, corner[1] * h] for corner in corners_norm],
            dtype=np.float32,
        )
        dst = np.array(
            [[0.0, 0.0], [field_width, 0.0],
             [field_width, field_height], [0.0, field_height]],
            dtype=np.float32,
        )
        self.H_img2cm = cv2.getPerspectiveTransform(src, dst)

    def _img_to_cm(self, x: int, y: int) -> Optional[Tuple[float, float]]:
        if self.H_img2cm is None:
            return None
        point = np.array([[[float(x), float(y)]]], dtype=np.float32)
        out = cv2.perspectiveTransform(point, self.H_img2cm)[0][0]
        return float(out[0]), float(out[1])

    def _ball_in_goal_roi(self, x, y, r, side, frame_shape) -> bool:
        h, w = frame_shape[:2]
        goal = self.goals_cfg.left if side == "left" else self.goals_cfg.right
        x0 = int(goal.x0 * w)
        y0 = int(goal.y0 * h)
        x1 = int(goal.x1 * w)
        y1 = int(goal.y1 * h)
        xmin, xmax = sorted((x0, x1))
        ymin, ymax = sorted((y0, y1))
        return (
            (x + r >= xmin) and (x - r <= xmax)
            and (y + r >= ymin) and (y - r <= ymax)
        )

    def _ball_zone(self, x: int, frame_shape) -> int | None:
        if self.field_cfg is None or self.field_cfg.rods is None:
            return None

        _, w = frame_shape[:2]
        x_norm = x / w
        rods = sorted(self.field_cfg.rods.xs_norm)
        for i, rod_x in enumerate(rods):
            if x_norm < rod_x:
                return i
        return len(rods)

    @property
    def speed_display(self):
        return self._speed_display
