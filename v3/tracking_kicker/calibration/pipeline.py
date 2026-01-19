import datetime
import numpy as np
from typing import Any, Dict, List

from .colors import ColorCalibrator
from .rods import RodCalibrator
from .goals import GoalCalibrator


class CalibrationPipeline:
    def __init__(self, rod_layout: List[Dict[str, Any]]):
        self.rod_layout = rod_layout
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

        rods_cfg = [{"x_norm": rods_norm[i], **self.rod_layout[i]} for i in range(8)]

        return {
            "created": datetime.datetime.now().isoformat(),
            "colors": colors_cfg,
            "rods": rods_cfg,
            "goals": goals_cfg
        }
