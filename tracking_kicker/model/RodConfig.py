from dataclasses import dataclass
from typing import List


@dataclass
class RodConfig:
    """
    Vertikale Stangen, normierte x-Positionen [0..1]
    Von links nach rechts sortiert
    """
    xs_norm: List[float]  # Länge = 8
