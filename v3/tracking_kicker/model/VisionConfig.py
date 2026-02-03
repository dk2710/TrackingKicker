from tracking_kicker.model.Vision import GoalsConfig, FieldConfig
from dataclasses import dataclass

@dataclass
class VisionConfig:
    goals: GoalsConfig
    field: FieldConfig | None = None
