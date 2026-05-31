from __future__ import annotations

from dataclasses import dataclass
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import TendonEffortActionCfg, TendonEffortAction
from mjlab.actuator.actuator import TransmissionType

@dataclass
class MyoMuscleActivationActionCfg(TendonEffortActionCfg):
    def __post_init__(self):
        self.transmission_type = TransmissionType.TENDON

    def build(self, env: ManagerBasedRlEnv) -> MyoMuscleActivationAction:
        return MyoMuscleActivationAction(self, env)


class MyoMuscleActivationAction(TendonEffortAction):
    """Control tendons via effort targets."""

    def __init__(self, cfg: MyoMuscleActivationActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)

    def process_actions(self, actions: torch.Tensor) -> None:
        # Apply the same sigmoid normalization as WalkEnvV0 / base_v0.py (CPU):
        #   ctrl = 1 / (1 + exp(-5 * (a - 0.5)))  =  sigmoid(5 * (a - 0.5))
        # Maps policy output [-1, 1] → muscle activation [~0, ~1].
        # Example: a=0 → 0.924, a=-1 → 0.076  (NOT the linear 0.5*(a+1) midpoint mapping).
        # Matches base_v0.py:90-92 which runs when normalize_act=True and muscles are present.
        self._raw_actions[:] = actions
        self._processed_actions = torch.sigmoid(5.0 * (self._raw_actions - 0.5))
