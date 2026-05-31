# MyoUser MJLab

Custom MyoUser task and vision components for running [MJLab](https://mujocolab.github.io/mjlab/main/index.html) training and playback through `uv`.

The wrapper registers a single MJLab task named `MyoUser`. The selected MyoUser task and vision components are configured through Hydra before execution is forwarded to MJLab.

## Setup

Create and sync the [uv](https://github.com/astral-sh/uv) environment:

```bash
uv sync
```

## Usage

Train with the default task:

```bash
uv run myo_train \
  task=universal \
  task.reach.dwell_continuous=True \
  rl.max_iterations=50
```

Play a trained run:

```bash
uv run myo_play -- \
  --viewer viser \
  --wandb-run-path ingrim4-universit-t-leipzig/mjlab/ut38fm1d
```

## Argument forwarding

`myo_train` and `myo_play` split command-line arguments into two groups:

```bash
uv run myo_train <myouser/hydra args> -- <mjlab args>
```

Arguments before `--` are handled by the MyoUser/Hydra configuration layer.

Arguments after `--` are forwarded to the underlying MJLab `train` or `play` command.

The MJLab task id is `MyoUser` and is inserted automatically if it is not already present.

Example:

```bash
uv run myo_play task=universal vision=disabled -- \
  --wandb-run-path ingrim4-universit-t-leipzig/mjlab/ut38fm1d
```

## Configuration

The main config selects a task and a vision component:

```yaml
defaults:
  - _self_
  - task: universal
  - vision: disabled
```

Common overrides:

```bash
uv run myo_train \
  task=universal \
  vision=disabled \
  env.num_envs=4096 \
  rl.max_iterations=250
```

## Registering tasks

Tasks are registered with `@myo_register_task`.

```python
@myo_register_task("universal")
class UniversalTaskComponent(myo.MyoComponent):
    def __init__(self, cfg: UniversalTaskConfig):
        self.cfg = cfg

    def modify_env_cfg(self, cfg: ManagerBasedRlEnvCfg, play: bool) -> None:
        pass
```

The registered class must:

* inherit from `myo.MyoComponent`
* define a constructor with only `self` and one typed config argument
* use a config type compatible with the MyoUser task config base type

The typed constructor argument is used as the Hydra config schema for the task. For example, `cfg: UniversalTaskConfig` registers `UniversalTaskConfig` as the Hydra config for `task=universal`.

The constructor parameter name is arbitrary, but the type annotation is required because the registry uses it to infer the task config type.

## Registering vision components

Vision components use the same pattern with `@myo_register_vision`.

```python
@myo_register_vision("disabled")
class DisabledVisionComponent(myo.MyoComponent):
    def __init__(self, cfg: DisabledVisionConfig):
        self.cfg = cfg

    def modify_rl_cfg(self, cfg: RslRlOnPolicyRunnerCfg) -> None:
        cfg.actor.class_name = "MLPModel"
        cfg.obs_groups = {
            "actor": ("agent_state", "task_state"),
            "critic": ("agent_state", "task_state"),
        }
```

The typed constructor argument is used as the Hydra config schema for the vision option. For example, `cfg: DisabledVisionConfig` registers `DisabledVisionConfig` as the Hydra config for `vision=disabled`.

## Importing registrations

Task and vision registrations happen when the Python modules containing the decorators are imported.

Each task or vision component must therefore be imported from the corresponding package `__init__.py`.

Example:

```python
# myo_core/task/universal/__init__.py
from .universal_task_config import UniversalTaskConfig
from .universal_task_component import UniversalTaskComponent
```

```python
# myo_core/task/__init__.py
from .universal import UniversalTaskConfig, UniversalTaskComponent
```

The same rule applies to vision components:

```python
# myo_core/vision/disabled/__init__.py
from .disabled_vision_config import DisabledVisionConfig
from .disabled_vision_component import DisabledVisionComponent
```

```python
# myo_core/vision/__init__.py
from .disabled import DisabledVisionConfig, DisabledVisionComponent
```

If a component module is not imported, its decorator is never executed and the component will not be available through Hydra.

## Repository layout

```text
myo_config/   # Hydra config files
myo_core/     # startup, registries, shared components, task/vision implementations
myo_user/     # assets and environment resources
```
