from dataclasses import dataclass
from ..vision_config import VisionConfig

@dataclass
class DisabledVisionConfig(VisionConfig):
  pass
