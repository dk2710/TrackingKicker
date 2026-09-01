from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict


@dataclass
class CameraSettings:
    """All camera values which are applied after every camera reconnect."""

    # 120 FPS keeps the current 1100x650 ROI below the practical GigE limit.
    force_fps: float = 120.0
    width: int = 1100
    height: int = 650
    offset_x: int = 240
    offset_y: int = 266
    exposure_time: float = 4000.0
    gain: float = 10.0
    pixel_format: str = "BayerRG8"
    exposure_auto: str = "Off"
    gain_auto: str = "Off"

    # latest_image_only is best for a live tuning preview. OneByOne is
    # available when measuring whether the transport can deliver every frame.
    grab_strategy: str = "latest_image_only"
    max_num_buffer: int = 64

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
}


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

    ball_raw = raw.get("ball_detection", {})
    ball_values = {
        name: value
        for name, value in ball_raw.items()
        if name in {item.name for item in fields(BallDetectionSettings)}
    }
    # Tracker-style [H, S, V] arrays are also accepted when copying values
    # from v3/tracking_kicker/config.json into the tuner.
    if "hsv_low" in ball_raw:
        ball_values.update(dict(zip(
            ("hsv_low_h", "hsv_low_s", "hsv_low_v"),
            ball_raw["hsv_low"],
        )))
    if "hsv_high" in ball_raw:
        ball_values.update(dict(zip(
            ("hsv_high_h", "hsv_high_s", "hsv_high_v"),
            ball_raw["hsv_high"],
        )))
    ball = BallDetectionSettings(**ball_values)

    server = dict(DEFAULT_SERVER_SETTINGS)
    server.update(raw.get("server", {}))
    return camera, ball, server


def save_settings(
    path: Path,
    camera: CameraSettings,
    ball: BallDetectionSettings,
    server: Dict[str, Any],
) -> None:
    """Persist the settings used by the tuner and by camera reconnects."""
    payload = {
        "server": {
            "host": str(server.get("host", DEFAULT_SERVER_SETTINGS["host"])),
            "port": int(server.get("port", DEFAULT_SERVER_SETTINGS["port"])),
        },
        "camera": asdict(camera),
        "ball_detection": asdict(ball),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)
