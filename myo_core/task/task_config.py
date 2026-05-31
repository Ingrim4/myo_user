from dataclasses import dataclass
from omegaconf import MISSING

from ..common import MyoComponentConfig

@dataclass
class TaskConfig(MyoComponentConfig):
    model_path: str = MISSING
