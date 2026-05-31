from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
import torch

def time(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.data.time

def joint_qpos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    jnt_ids = asset_cfg.joint_ids

    jnt_range = asset.data.joint_pos_limits[:, jnt_ids, :]
    qpos = asset.data.joint_pos[:, jnt_ids]

    qpos = (qpos - jnt_range[..., 0]) / (jnt_range[..., 1] - jnt_range[..., 0])
    qpos = (qpos - 0.5) * 2

    return qpos

def joint_qvel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    jnt_ids = asset_cfg.joint_ids

    qvel = asset.data.joint_vel[:, jnt_ids]

    return qvel

def joint_qacc(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    jnt_ids = asset_cfg.joint_ids

    qacc = asset.data.joint_acc[:, jnt_ids]

    return qacc

def act(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    act = (asset.data.data.act - 0.5) * 2

    return act

def site_pos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    site_ids = asset_cfg.site_ids

    sites_pos = asset.data.site_pos_w[:, site_ids]
    b, n, m = sites_pos.shape

    return sites_pos.reshape(b, n * m)