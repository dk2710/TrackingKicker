from dataclasses import dataclass

@dataclass
class Camera:
    # Kamera / FPS
    force_fps: int #= 60

    # ROI
    height: int #= 650
    offset_y: int #= 266
    width: int #= 1100
    offset_x: int #= 240

    # Exposure / Gain
    exposure_time: float #= 4000.0
    gain: float

    # Debug / Modus
    grab_strategy_latest_only: bool #= True

    k1: float = 0.00
    k2: float = 0.05
    k3: float = 0.00
    p1: float = 0.00
    p2: float = 0.00
    scale: float = 1.00  # wichtig: 1.0, nicht 0.0
