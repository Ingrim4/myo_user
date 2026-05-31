from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
import torch

from .observation import joint_qacc

def neural_effort(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    ctrl = asset.data.tendon_effort_target

    ctrl_magnitude = torch.linalg.vector_norm(ctrl, dim=-1)
    neural_effort = -1.0 * (ctrl_magnitude ** 2)

    return neural_effort

def jac_effort(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    ctrl = asset.data.tendon_effort_target

    r_effort = 0.00198 * torch.linalg.vector_norm(ctrl, dim=-1) ** 2
    r_jacc = 6.67e-6 * torch.linalg.vector_norm(joint_qacc(env, asset_cfg), dim=-1) ** 2 
    effort_cost = -(r_effort + r_jacc)

    return effort_cost
