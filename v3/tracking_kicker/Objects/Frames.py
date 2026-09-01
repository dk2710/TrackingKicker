from typing import Optional, Tuple
import numpy as np




class Frame:
    frame: np.ndarray
    timestamp: float

    ball: Optional[Tuple[float, float, float]]
    # True only when the detector connected this contour to the previous
    # ball track with a plausible time and movement interval.
    ball_track_valid: bool
    ball_track_reason: str
    ball_track_distance_px: Optional[float]
