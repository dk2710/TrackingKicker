from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class GoalROI:
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class GoalsConfig:
    left: GoalROI
    right: GoalROI


@dataclass
class FieldSize:
    w: float
    h: float


@dataclass
class FieldConfig:
    size_cm: FieldSize
    corners_norm: List[Tuple[float, float]]
