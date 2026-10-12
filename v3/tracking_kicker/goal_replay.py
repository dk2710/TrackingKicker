"""Pure helpers for the three-second goal replay.

The camera continues to deliver frames at its configured high rate.  Goal
media is deliberately reduced only when an event is created: the API video
contains three seconds at 30 FPS, while the retro analysis reads the original
timestamps from the final second.
"""

from __future__ import annotations

from bisect import bisect_left
import math
from typing import Sequence

import numpy as np


ReplayItem = tuple[np.ndarray, float]


def select_goal_frames(
    source_frames: Sequence[ReplayItem],
    goal_timestamp: float,
    *,
    duration_seconds: float = 3.0,
    output_fps: float = 30.0,
) -> list[ReplayItem]:
    """Select an evenly timed 30-FPS window ending at the goal.

    The source normally contains about 540 camera frames.  Selecting by
    timestamps instead of list positions also works when the transport has
    dropped a frame.  The returned list always has the requested number of
    output frames when the source contains at least one frame in the window;
    the nearest available source frame is reused only for a missing timestamp.
    """

    frame_count = int(round(duration_seconds * output_fps))
    if frame_count <= 0:
        return []

    end = float(goal_timestamp)
    start = end - float(duration_seconds)
    window = sorted(
        (
            (image, float(timestamp))
            for image, timestamp in source_frames
            if start <= float(timestamp) <= end
        ),
        key=lambda item: item[1],
    )
    if not window:
        return []

    timestamps = [item[1] for item in window]
    # A 30-FPS, three-second stream has timestamps 0 .. 2.9666s relative to
    # its start.  The next frame boundary is the goal timestamp itself.
    targets = start + np.arange(frame_count, dtype=float) / float(output_fps)
    selected: list[ReplayItem] = []
    for target in targets:
        index = bisect_left(timestamps, float(target))
        if index <= 0:
            selected.append(window[0])
            continue
        if index >= len(window):
            selected.append(window[-1])
            continue

        before = window[index - 1]
        after = window[index]
        selected.append(
            before if target - before[1] <= after[1] - target else after
        )

    return selected


def select_analysis_window(
    source_frames: Sequence[ReplayItem],
    goal_timestamp: float,
    *,
    duration_seconds: float = 1.0,
) -> list[ReplayItem]:
    """Return the original-rate frames immediately before the goal."""

    end = float(goal_timestamp)
    start = end - float(duration_seconds)
    return sorted(
        (
            (image, float(timestamp))
            for image, timestamp in source_frames
            if start <= float(timestamp) <= end
        ),
        key=lambda item: item[1],
    )


def calculate_three_point_speed_kmh(points: Sequence[dict[str, float]]) -> float:
    """Calculate speed from exactly three timestamped world positions.

    A least-squares line through all three points uses both motion segments
    and avoids choosing an arbitrary single pair.  World coordinates are in
    centimetres, so centimetres per second are converted to km/h by 0.036.
    """

    if len(points) != 3:
        return 0.0

    times = np.asarray([float(point["timestamp"]) for point in points])
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        return 0.0

    x = np.asarray([float(point["x_cm"]) for point in points])
    y = np.asarray([float(point["y_cm"]) for point in points])
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return 0.0

    relative_time = times - times[0]
    design = np.column_stack((relative_time, np.ones(3)))
    velocity_x = float(np.linalg.lstsq(design, x, rcond=None)[0][0])
    velocity_y = float(np.linalg.lstsq(design, y, rcond=None)[0][0])
    speed = math.hypot(velocity_x, velocity_y) * 0.036
    return float(speed) if math.isfinite(speed) else 0.0
