"""Configuration shared by the physical camera source and the image pipeline."""

from dataclasses import dataclass


@dataclass
class Camera:
    """Application-side camera settings.

    The values in this class are deliberately independent of pypylon.  This
    keeps configuration loading and replay mode usable on machines where
    pylon is not installed.
    """

    # Acquisition
    force_fps: int

    # ROI
    height: int
    offset_y: int
    width: int
    offset_x: int

    # Exposure / gain
    exposure_time: float
    gain: float

    # Runtime behavior
    grab_strategy_latest_only: bool

    # Optional stable device selection and host buffer configuration
    serial_number: str | None = None
    max_num_buffer: int = 256
    pixel_format: str = "BayerRG8"

    # Lens correction
    k1: float = 0.00
    k2: float = 0.05
    k3: float = 0.00
    p1: float = 0.00
    p2: float = 0.00
    scale: float = 1.00
