from v4.tracking_kicker.model.RodConfig import RodConfig
from v4.tracking_kicker.model.Vision import GoalsConfig, FieldConfig, GoalROI, FieldSize
from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class VisionConfig:
    goals: GoalsConfig
    field: FieldConfig | None = None

    # 🔒 zentraler Standard-Pfad
    DEFAULT_PATH = Path(__file__).resolve().parents[1] / "vision_config.json"

    # ---------- JSON ----------
    @classmethod
    def load(cls, path: Path | None = None) -> "VisionConfig":
        path = path or cls.DEFAULT_PATH

        if not path.exists():
            raise RuntimeError(f"❌ VisionConfig fehlt: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        goals_raw = data["goals"]
        field_raw = data.get("field")

        return cls(
            goals=GoalsConfig(
                left=GoalROI(**goals_raw["left"]),
                right=GoalROI(**goals_raw["right"]),
            ),
            field=FieldConfig(
                size_cm=FieldSize(**field_raw["size_cm"]),
                corners_norm=field_raw["corners_norm"],
                rods=RodConfig(**field_raw["rods"]) if "rods" in field_raw else None
            ) if field_raw else None
        )

    def save(self, path: Path | None = None) -> None:
        path = path or self.DEFAULT_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    def reload(self):
        new = VisionConfig.load(self.DEFAULT_PATH)

        self.goals = new.goals
        self.field = new.field