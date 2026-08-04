from typing import Optional, Tuple
import numpy as np




class Frame:
    frame: np.ndarray
    timestamp: float

    ball: Optional[Tuple[float, float, float]]
