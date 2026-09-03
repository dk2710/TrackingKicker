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
    # LatestImageOnly already drops stale frames, so a very large host pool
    # only adds memory pressure during high-rate acquisition.
    max_num_buffer: int = 32
    pixel_format: str = "BayerRG8"

    # pylon USB stream transport.  These are the values recommended by
    # Basler for high-throughput USB3 acquisition and are kept configurable
    # because the usable limits depend on the host controller.
    # Moderate DMA transfers are more stable on the RK3588 USB host. Their
    # product stays comfortably below Linux' default usbfs memory budget.
    max_transfer_size: int = 262144
    num_max_queued_urbs: int = 16
    # This priority setting belongs to pylon's socket driver and is not
    # available for USB cameras.
    receive_thread_priority_override: bool = False
    receive_thread_priority: int = 15
    # Smooth the USB traffic without lowering 180 FPS. The configured ROI
    # needs about 129 MB/s of BayerRG8 payload at 180 FPS; 150 MB/s leaves
    # headroom for USB3 Vision overhead while avoiding a large burst.
    device_link_throughput_limit: int = 150_000_000

    # Optionally reserve host CPUs for the pylon receive process. The process
    # is pinned only after the camera has opened, so affinity cannot interfere
    # with pylon's device initialization.
    capture_cpu: int | list[int] | None = 7
    # Keep pylon's USB receive threads free to use all available CPUs by
    # default. Pinning the complete process to one core can starve the USB3
    # transport at high frame rates; enable it only for a measured setup.
    reserve_capture_cpu: bool = False

    # Lens correction
    k1: float = 0.00
    k2: float = 0.05
    k3: float = 0.00
    p1: float = 0.00
    p2: float = 0.00
    scale: float = 1.00
