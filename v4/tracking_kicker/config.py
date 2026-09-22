from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import os
import json
from typing import Any, Dict
from v4.tracking_kicker.model.Camera import Camera


@dataclass
class BallDetectionSettings:
    """The v3 detector/API configuration carried into v4."""

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


@dataclass
class AppConfig:
    camera: Camera | None = None
    ball_detection: Dict[str, Any] = field(default_factory=dict)
    headless: bool = True
    ball_detection_enabled: bool = True
    replay_seconds: int = 3  # retained for the v3 API contract; unused in v4
    target_display_fps: int = 30
    # The camera is drained at full rate. The value can be raised to reduce
    # detector work while keeping the complete camera count visible.
    detection_frame_stride: int = 2
    use_camera: bool = True
    server: Dict[str, Any] = field(default_factory=lambda: {
        "host": "0.0.0.0",
        "port": 5000,
    })

    # === Paths ===
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[0])
    config_path: Path = field(init=False)
    videos_dir: Path = field(init=False)

    def __post_init__(self):
        self.videos_dir = self.base_dir / "videos"
        self.config_path = self.base_dir / "config.json"
        self.videos_dir.mkdir(parents=True, exist_ok=True)


    # ---------- JSON ----------
    @classmethod
    def load(cls, path: Path | None = None) -> "AppConfig":
        tmp = cls()  # nur um base_dir zu kennen
        cfg_path = path or tmp.config_path

        if not cfg_path.exists():
            return cls.from_env()

        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cam_data = data.pop("camera", None)
        camera = Camera(**cam_data) if cam_data else None
        ball_detection = data.pop("ball_detection", {})
        server = data.pop("server", None) or {
            "host": "0.0.0.0",
            "port": 5000,
        }
        valid_names = {item.name for item in fields(cls) if item.init}
        values = {name: value for name, value in data.items() if name in valid_names}
        return cls(
            camera=camera,
            ball_detection=ball_detection,
            server=server,
            **values,
        )

    def save(self, path: Path | None = None):
        path = path or self.config_path
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def from_env() -> "AppConfig":
        return AppConfig(
            headless=os.environ.get("HEADLESS", "0") == "1"
        )

    def to_dict(self) -> Dict[str, Any]:
        values = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.init and f.name != "base_dir"
        }
        if self.camera is not None:
            values['camera'] = asdict(self.camera)
        return values
