from dataclasses import dataclass


@dataclass
class Camera:
    """Camera configuration shared with the former v3 API."""

    # 0 means: do not cap the camera; use the camera's resulting rate.
    force_fps: float = 0.0
    height: int = 650
    offset_y: int = 266
    width: int = 1100
    offset_x: int = 240
    exposure_time: float = 4000.0
    gain: float = 10.0
    grab_strategy_latest_only: bool = False

    # v3 configuration fields retained for the HTTP/API contract. v4 uses
    # direct_basler and never creates a shared-memory frame transport.
    serial_number: str | None = "40719659"
    pixel_format: str = "BayerRG8"
    exposure_auto: str = "Off"
    gain_auto: str = "Off"
    acquisition_frame_rate_enable: bool = False
    source: str = "direct_basler"
    max_num_buffer: int = 0
    max_transfer_size: int = 0
    max_buffer_size: int = 0
    num_max_queued_urbs: int = 0
    transfer_loop_thread_priority: int = 0
    handoff_queue_size: int = 1
    handoff_frame_stride: int = 2
    capture_cpu: int | None = 0
    reserve_capture_cpu: bool = True
    device_link_throughput_limit_mode: str = "Off"
    device_link_throughput_limit: int = 0
    packet_size: int = 0
    inter_packet_delay: int = 0

    k1: float = 0.00
    k2: float = 0.05
    k3: float = 0.00
    p1: float = 0.00
    p2: float = 0.00
    scale: float = 1.00
