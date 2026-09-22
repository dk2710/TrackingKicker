import datetime
import numpy as np
from typing import Any, Dict, List
from .goals import GoalCalibrator
from .field import FieldCalibrator

class CalibrationPipeline:
    def __init__(self):
        self.field_cal = FieldCalibrator()
        self.goal_cal = GoalCalibrator()

    def run(self, first_frame: np.ndarray) -> Dict[str, Any]:
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


        return {
            "created": datetime.datetime.now().isoformat(),
            "goals": goals_cfg,
            "field": field_cfg
        }
