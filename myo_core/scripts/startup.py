from typing import Sequence

import sys
from collections.abc import Callable
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from myo_core import MyoConfig, myo_mjlab_register

# Only uncomment for debug (20% slow-down)
#import os
#os.environ['CUDA_LAUNCH_BLOCKING']="1"
#os.environ['TORCH_USE_CUDA_DSA'] = "1"

def split_args(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """
    Split command line into:
      - Hydra overrides before --
      - MJLab/tyro args after --

    Example:
      task=universal vision=yolo -- MyoUser --max-iterations 500
    """
    argv = list(argv)

    if "--" not in argv:
        return argv, []

    sep = argv.index("--")
    return argv[:sep], argv[sep + 1:]

def register_myo_from_cfg(cfg: DictConfig) -> None:
    myo_cfg: MyoConfig = OmegaConf.to_object(cfg)
    myo_mjlab_register(myo_cfg)

def startup(mjlab_main: Callable[[], int | None]) -> None:
    sys.argv[0] = sys.argv[0].removesuffix(".exe")

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

    register_myo_from_cfg(task_cfg)

    if "MyoUser" not in mjlab_args:
        mjlab_args = ["MyoUser", *mjlab_args]

    sys.argv = [sys.argv[0], *mjlab_args]
    exit_code = mjlab_main()

    sys.exit(0 if exit_code is None else exit_code)