from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

@dataclass
class MyoComponentConfig:
  pass

class MyoComponent:
  def modify_env_cfg(self, cfg: ManagerBasedRlEnvCfg, play: bool) -> None:
    """Mutate the environment config in place.

    Args:
      cfg: Environment config to modify.
      play: Whether the config is being prepared for train/play mode.
    """
    pass

  def modify_rl_cfg(self, cfg: RslRlOnPolicyRunnerCfg) -> None:
    """Mutate the RL runner config in place."""
    pass