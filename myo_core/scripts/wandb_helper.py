from __future__ import annotations

from typing import Any

import os
import sys

from omegaconf import DictConfig, OmegaConf
from myo_core import MyoConfig


def patch_wandb(
    *,
    full_argv: list[str],
    launch_command: str,
    hydra_cfg: DictConfig,
    myo_cfg: MyoConfig,
) -> None:
    """
    Patch wandb.init so MJLab runs log the full launch command and Hydra config.

    The patch preserves the real command-line arguments during wandb initialization,
    injects the resolved Myo/Hydra config into the W&B config, and disables W&B
    when configured to do so.
    """
    if not myo_cfg.wandb.enabled:
        os.environ["WANDB_MODE"] = "disabled"

    try:
        import wandb
    except ImportError:
        return

    if getattr(wandb.init, "_myo_patched", False):
        return

    original_init = wandb.init
    resolved_cfg = to_wandb_config(hydra_cfg)

    def init_with_full_argv_and_config(*args, **kwargs):
        """
        Replacement for wandb.init that adds Myo config and restores full argv.
        """
        config_arg = kwargs.pop("config", None)

        if config_arg is None:
            config: dict[str, Any] = {}
        elif isinstance(config_arg, dict):
            config = config_arg
        else:
            config = {"mjlab": str(config_arg)}

        config["launch_command"] = launch_command
        config["myo"] = resolved_cfg

        kwargs["config"] = config
        kwargs["name"] = myo_cfg.wandb.name

        if not myo_cfg.wandb.enabled:
            kwargs["mode"] = "disabled"

        old_argv = sys.argv[:]
        try:
            sys.argv = full_argv[:]
            run = original_init(*args, **kwargs)
        finally:
            sys.argv = old_argv

        return run

    init_with_full_argv_and_config._myo_patched = True
    wandb.init = init_with_full_argv_and_config


def to_wandb_config(cfg: DictConfig) -> dict[str, Any]:
    """
    Convert an OmegaConf config into a plain JSON-compatible dictionary.

    Interpolations are resolved eagerly. Non-primitive values are converted to
    strings so they can safely be stored in W&B config.
    """
    data = OmegaConf.to_container(
        cfg,
        resolve=True,
        throw_on_missing=True,
        enum_to_str=True,
    )

    def make_jsonable(x: Any) -> Any:
        """
        Recursively convert nested values into W&B-safe primitives.
        """
        if isinstance(x, dict):
            return {str(k): make_jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [make_jsonable(v) for v in x]
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        return str(x)

    return {str(k): make_jsonable(v) for k, v in data.items()}
