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


DEFAULT_SERVER_SETTINGS = {
    "host": "0.0.0.0",
    "port": 5010,
}


def load_settings(path: Path) -> tuple[CameraSettings, Dict[str, Any]]:
    """Load camera and web-server settings, filling missing values safely."""
    if not path.exists():
        return CameraSettings(), dict(DEFAULT_SERVER_SETTINGS)

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

    server = dict(DEFAULT_SERVER_SETTINGS)
    server.update(raw.get("server", {}))
    return camera, server


def save_settings(
    path: Path,
    camera: CameraSettings,
    server: Dict[str, Any],
) -> None:
    """Persist the settings used by the tuner and by camera reconnects."""
    payload = {
        "server": {
            "host": str(server.get("host", DEFAULT_SERVER_SETTINGS["host"])),
            "port": int(server.get("port", DEFAULT_SERVER_SETTINGS["port"])),
        },
        "camera": asdict(camera),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)
