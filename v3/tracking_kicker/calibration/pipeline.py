import datetime
import numpy as np
from typing import Any, Dict, List

from .colors import ColorCalibrator
from .rods import RodCalibrator
from .goals import GoalCalibrator
from .field import FieldCalibrator

ROD_LAYOUT: List[Dict[str, Any]] = [
    {"team": "black", "expected": 1},
    {"team": "black", "expected": 2},
    {"team": "white", "expected": 2},
    {"team": "black", "expected": 5},
    {"team": "white", "expected": 5},
    {"team": "black", "expected": 3},
    {"team": "white", "expected": 2},
    {"team": "white", "expected": 1},
]

class CalibrationPipeline:
    def __init__(self):
        self.field_cal = FieldCalibrator()
        self.color_cal = ColorCalibrator()
        self.rod_cal = RodCalibrator()
        self.goal_cal = GoalCalibrator()

    def run(self, first_frame: np.ndarray) -> Dict[str, Any]:
        colors_cfg = self.color_cal.calibrate(first_frame)

        rods_norm = self.rod_cal.calibrate(first_frame, n_rods=8)
        if rods_norm is None or len(rods_norm) != 8:
            raise RuntimeError("❌ Rod-Kalibrierung abgebrochen oder unvollständig.")

        goals_cfg = self.goal_cal.calibrate(first_frame)
        if goals_cfg is None:
            raise RuntimeError("❌ Tor-Kalibrierung abgebrochen oder unvollständig.")

        field_corners_norm = self.field_cal.calibrate(first_frame)
        if field_corners_norm is None or len(field_corners_norm) != 4:
            raise RuntimeError("❌ Spielfeld-Ecken Kalibrierung abgebrochen oder unvollständig.")

        field_cfg = {
            "size_cm": {"w": 120.0, "h": 68.0},
            "corners_norm": field_corners_norm  # [OL, OR, UR, UL]
        }

        rods_cfg = [{"x_norm": rods_norm[i], **ROD_LAYOUT[i]} for i in range(8)]

        return {
            "created": datetime.datetime.now().isoformat(),
            "colors": colors_cfg,
            "rods": rods_cfg,
            "goals": goals_cfg,
            "field": field_cfg
        }

