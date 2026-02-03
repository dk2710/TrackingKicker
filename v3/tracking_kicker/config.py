from dataclasses import dataclass, field
from pathlib import Path
import os
import json
from typing import Any, Dict
from dataclasses import asdict, fields
from v3.tracking_kicker.model.Camera import Camera
from tracking_kicker.model.Vision import GoalsConfig, FieldConfig, GoalROI, FieldSize
from tracking_kicker.model.VisionConfig import VisionConfig


@dataclass
class AppConfig:
    camera: Camera | None = None
    # === Runtime ===
    headless: bool = False
    replay_seconds: int = 3

    # === Video ===
    target_display_fps: int = 60
    replay_fps: int = 120
    use_camera: bool = True
    debug_video: str | None = None

    vision: VisionConfig | None = None

    # === Paths ===
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
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

        # Camera
        cam_data = data.pop("camera", None)
        camera = Camera(**cam_data) if cam_data else None

        # Vision
        goals_raw = data.pop("goals", None)
        field_raw = data.pop("field", None)

        vision = None
        if goals_raw:
            vision = VisionConfig(
                goals=GoalsConfig(
                    left=GoalROI(**goals_raw["left"]),
                    right=GoalROI(**goals_raw["right"]),
                ),
                field=FieldConfig(
                    size_cm=FieldSize(**field_raw["size_cm"]),
                    corners_norm=field_raw["corners_norm"],
                ) if field_raw else None
            )

        # 🔒 FILTER
        allowed = {f.name for f in fields(cls) if f.init}
        data = {k: v for k, v in data.items() if k in allowed}

        return cls(camera=camera, vision=vision, **data)

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
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.init  # nur Felder, die aus JSON kommen dürfen
        }

