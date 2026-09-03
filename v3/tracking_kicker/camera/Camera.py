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

    # pylon USB stream transport.  These are the values recommended by
    # Basler for high-throughput USB3 acquisition and are kept configurable
    # because the usable limits depend on the host controller.
    max_transfer_size: int = 1048568
    num_max_queued_urbs: int = 64
    receive_thread_priority_override: bool = True
    receive_thread_priority: int = 15

    # Reserve one host CPU for the pylon receive process.  The process is
    # pinned only after the camera has opened, so CPU affinity cannot interfere
    # with pylon's device initialization.
    capture_cpu: int | None = 7
    reserve_capture_cpu: bool = True

    # Lens correction
    k1: float = 0.00
    k2: float = 0.05
    k3: float = 0.00
    p1: float = 0.00
    p2: float = 0.00
    scale: float = 1.00
