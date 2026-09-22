from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict


@dataclass
class CameraSettings:
    """All camera values which are applied after every camera reconnect."""

    # The web stream uses 30 FPS. Higher acquisition rates remain available,
    # but the USB transport must be allowed to run lower when necessary.
    force_fps: float = 60.0
    # Keep the camera's acquisition-rate limiter disabled unless explicitly
    # requested. ``force_fps`` remains the nominal/expected rate for status
    # and detection timing; it is not written to the camera while disabled.
    acquisition_frame_rate_enable: bool = False
    width: int = 1100
    height: int = 650
    offset_x: int = 240
    offset_y: int = 266
    exposure_time: float = 4000.0
    gain: float = 10.0
    pixel_format: str = "BayerRG8"
    exposure_auto: str = "Off"
    gain_auto: str = "Off"

    # LatestImageOnly is the production strategy. It prevents an application
    # backlog when the tracker or preview temporarily needs more time.
    grab_strategy: str = "latest_image_only"
    # v4 leaves the pylon default untouched.
    max_num_buffer: int = 0

    # Zero leaves the pylon/USB stream-grabber defaults untouched, matching v4.
    max_transfer_size: int = 0
    num_max_queued_urbs: int = 0

    # Zero leaves the camera's default throughput handling untouched, matching v4.
    device_link_throughput_limit_mode: str = "Off"
    device_link_throughput_limit: int = 0

    # Optional GigE transport settings. Zero means leave the camera default.
    packet_size: int = 0
    inter_packet_delay: int = 0


@dataclass
class BallDetectionSettings:
    """HSV and contour filters used by the tracker's ball detector."""

    hsv_low_h: int = 100
    hsv_low_s: int = 110
    hsv_low_v: int = 90
    hsv_high_h: int = 112
    hsv_high_s: int = 255
    hsv_high_v: int = 255
    area_min: float = 60.0
    area_max: float = 12000.0
    detection_scale: float = 0.5
    radius_min_px: float = 5.0
    radius_max_px: float = 25.0
    circularity_min: float = 0.5
    solidity_min: float = 0.85

    def as_tracker_dict(self) -> Dict[str, Any]:
        return {
            "hsv_low": [self.hsv_low_h, self.hsv_low_s, self.hsv_low_v],
            "hsv_high": [self.hsv_high_h, self.hsv_high_s, self.hsv_high_v],
            "area_min": self.area_min,
            "area_max": self.area_max,
            "detection_scale": self.detection_scale,
            "radius_min_px": self.radius_min_px,
            "radius_max_px": self.radius_max_px,
            "circularity_min": self.circularity_min,
            "solidity_min": self.solidity_min,
        }


DEFAULT_SERVER_SETTINGS = {
    "host": "0.0.0.0",
    "port": 5010,
    "camera_control_url": "http://127.0.0.1:5011",
    "ball_detection_enabled": False,
    "frame_ipc_name": "tracking_kicker_camera_frames",
    "frame_ipc_slots": 3,
}


def _ball_settings_from_mapping(values: Dict[str, Any]) -> BallDetectionSettings:
    """Read both tuner-style scalar values and tracker-style HSV arrays."""
    valid_names = {item.name for item in fields(BallDetectionSettings)}
    ball_values = {
        name: value for name, value in values.items() if name in valid_names
    }
    if "hsv_low" in values:
        ball_values.update(dict(zip(
            ("hsv_low_h", "hsv_low_s", "hsv_low_v"), values["hsv_low"]
        )))
    if "hsv_high" in values:
        ball_values.update(dict(zip(
            ("hsv_high_h", "hsv_high_s", "hsv_high_v"), values["hsv_high"]
        )))
    return BallDetectionSettings(**ball_values)


def load_tracker_ball_settings(
    path: Path,
) -> tuple[BallDetectionSettings, bool]:
    """Load ball settings and enabled state from the tracker config."""
    if not path.exists():
        return BallDetectionSettings(), False
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return (
        _ball_settings_from_mapping(raw.get("ball_detection", {})),
        bool(raw.get("ball_detection_enabled", False)),
    )


def save_tracker_ball_settings(
    path: Path,
    ball: BallDetectionSettings,
    enabled: bool,
) -> None:
    """Persist ball tuning in the tracker configuration atomically."""
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    else:
        document = {}
    document["ball_detection"] = ball.as_tracker_dict()
    document["ball_detection_enabled"] = bool(enabled)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


def load_settings(
    path: Path,
) -> tuple[CameraSettings, BallDetectionSettings, Dict[str, Any]]:
    """Load camera and web-server settings, filling missing values safely."""
    if not path.exists():
        return CameraSettings(), BallDetectionSettings(), dict(DEFAULT_SERVER_SETTINGS)

    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    camera_raw = raw.get("camera", raw)
    valid_names = {item.name for item in fields(CameraSettings)}
    camera_values = {
        name: value
        for name, value in camera_raw.items()
        if name in valid_names
    }
    camera = CameraSettings(**camera_values)

    ball = _ball_settings_from_mapping(raw.get("ball_detection", {}))

    server = dict(DEFAULT_SERVER_SETTINGS)
    server.update(raw.get("server", {}))
    return camera, ball, server


def save_settings(
    path: Path,
    camera: CameraSettings,
    _ball: BallDetectionSettings,
    server: Dict[str, Any],
) -> None:
    """Persist server and camera settings.

    Ball detection is intentionally persisted in the tracker configuration;
    the ``_ball`` argument remains for compatibility with existing callers.
    """
    payload = {
        "server": {
            "host": str(server.get("host", DEFAULT_SERVER_SETTINGS["host"])),
            "port": int(server.get("port", DEFAULT_SERVER_SETTINGS["port"])),
            "camera_control_url": str(
                server.get(
                    "camera_control_url",
                    DEFAULT_SERVER_SETTINGS["camera_control_url"],
                )
            ),
            "ball_detection_enabled": bool(
                server.get(
                    "ball_detection_enabled",
                    DEFAULT_SERVER_SETTINGS["ball_detection_enabled"],
                )
            ),
            "frame_ipc_name": str(
                server.get(
                    "frame_ipc_name",
                    DEFAULT_SERVER_SETTINGS["frame_ipc_name"],
                )
            ),
            "frame_ipc_slots": max(
                3,
                int(
                    server.get(
                        "frame_ipc_slots",
                        DEFAULT_SERVER_SETTINGS["frame_ipc_slots"],
                    )
                ),
            ),
        },
        "camera": asdict(camera),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)
