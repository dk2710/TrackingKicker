from dataclasses import dataclass, field
from pathlib import Path
import os
import json
from typing import Any, Dict
from dataclasses import asdict, fields
from .camera import Camera


@dataclass
class AppConfig:
    camera: Camera | None = None
    # Einstellungen fuer die farbbasierte Ballkontur-Erkennung.
    ball_detection: Dict[str, Any] = field(default_factory=dict)
    # === Runtime ===
    headless: bool = False
    ball_detection_enabled: bool = False
    replay_seconds: int = 3
    replay_ipc_name: str = "tracking_kicker_replay_frames"
    replay_ipc_slots: int = 160

    # === Video ===
    target_display_fps: int = 60
    use_camera: bool = True

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

        # Camera
        cam_data = data.pop("camera", None)
        camera = Camera(**cam_data) if cam_data else None
        ball_detection = data.pop("ball_detection", {})

        return cls(
            camera=camera,
            ball_detection=ball_detection,
            **data,
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
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.init  # nur Felder, die aus JSON kommen dürfen
        }

