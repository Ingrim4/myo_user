from typing import Sequence

import sys
import shlex
from collections.abc import Callable
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from myo_core import MyoConfig, myo_mjlab_register
from .wandb_helper import patch_wandb


# Only uncomment for debug (20% slow-down)
# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"


def split_args(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """
    Split command-line arguments into Hydra args and MJLab args.

    Arguments before `--` are treated as Hydra overrides.
    Arguments after `--` are forwarded to MJLab/tyro.

    Example:
        task=universal vision=yolo -- MyoUser --max-iterations 500
    """
    argv = list(argv)

    if "--" not in argv:
        return argv, []

    sep = argv.index("--")
    return argv[:sep], argv[sep + 1:]


def startup(mjlab_main: Callable[[], int | None]) -> None:
    """
    Initialize Hydra, register Myo components, patch W&B, and run MJLab.

    Hydra receives the arguments before `--`. MJLab receives the arguments after
    `--`, with `MyoUser` inserted automatically when missing.
    """
    sys.argv[0] = sys.argv[0].removesuffix(".exe")

    entrypoint = Path(sys.argv[0]).name
    hydra_args, mjlab_args = split_args(sys.argv[1:])

    root = Path(__file__).resolve().parents[2]
    config_dir = root / "myo_config"

    with initialize_config_dir(
        version_base=None,
        config_dir=str(config_dir),
        job_name="myo_startup",
    ):
        cfg = compose(
            config_name="base",
            overrides=hydra_args,
            return_hydra_config=True,
        )

    HydraConfig.instance().set_config(cfg)

    task_cfg = cfg.copy()
    with open_dict(task_cfg):
        task_cfg.pop("hydra", None)

    myo_cfg: MyoConfig = OmegaConf.to_object(task_cfg)
    myo_mjlab_register(myo_cfg)

    if "MyoUser" not in mjlab_args:
        mjlab_args = ["MyoUser", *mjlab_args]

    full_argv = [sys.argv[0], *hydra_args, "--", *mjlab_args]
    launch_command = shlex.join(["uv", "run", entrypoint, *hydra_args, "--", *mjlab_args])

    patch_wandb(
        full_argv=full_argv,
        launch_command=launch_command,
        hydra_cfg=task_cfg,
        myo_cfg=myo_cfg,
    )

    sys.argv = [sys.argv[0], *mjlab_args]
    exit_code = mjlab_main()

    sys.exit(0 if exit_code is None else exit_code)
