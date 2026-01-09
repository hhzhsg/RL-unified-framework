# VLA-RL AI Coding Guide

## Architecture Overview
**Modular RL framework** supporting Offline/Online/Human-in-the-Loop training with registry-based plugin system.

Training and inference run separately, synchronized via WeightSync queues. Training pulls batches from DataHub buffers and pushes weights; inference pulls weights and executes policy in env.

```
Intervention/Demo/Rollout → Sampler → TrainingLoop → Algorithm → ModelGroup
         ↑                                                          ↓ sync
      write                                                    InferenceLoop ↔ Env
```

## Registry Pattern (Core Extension Mechanism)
All components use **Base Class + Registry + Factory** pattern. To add new components:

| Component | Base Class | Registry | Config Field | Location |
|-----------|-----------|----------|--------------|----------|
| Algorithm | `BaseAlgorithm` | `ALGORITHM_REGISTRY` | `algorithm.name` | `algorithm/__init__.py` |
| Policy | `BasePolicy` | `POLICY_REGISTRY` | `model.policy_type` | `model/__init__.py` |
| Environment | `BaseEnv` | `ENV_REGISTRY` | `env.name` | `env/__init__.py` |
| Sampler | `BaseSampleStrategy` | `STRATEGY_REGISTRY` | `sample_strategy` | `buffer/sample_strategy.py` |

**Universal extension pattern:**
```python
# 1. Inherit base class
class MyAlgo(BaseAlgorithm):
    REQUIRED_MODELS = ["policy"]  # Declare dependencies
    def train_step(self, batch) -> Dict[str, float]: ...

# 2. Register (in module __init__.py)
ALGORITHM_REGISTRY["my_algo"] = MyAlgo

# 3. Use in config/train_config.yaml
algorithm:
  name: "my_algo"
  lr: 1e-4
```

## Key Components

- **DataHub** ([buffer/data_hub.py](buffer/data_hub.py)): Unified interface for 3 data sources. Demo=HDF5 read-only, Rollout=FIFO ring buffer, Intervention=with disk persistence.
- **Sampler** ([buffer/sample_strategy.py](buffer/sample_strategy.py)): `DemoOnlyStrategy`/`RolloutOnlyStrategy`/`MixedStrategy` control data mixing ratios.
- **ModelGroup** ([model/model_group.py](model/model_group.py)): Central model registry with freeze/unfreeze control. Naming convention: `policy`, `q1`/`q2`, `target_q1`/`target_q2`, `vf`.
- **TrainingLoop** ([core/training_loop.py](core/training_loop.py)): Multi-stage training via `Stage` config, auto freeze/unfreeze per stage, algorithm instance caching.
- **Algorithm** ([algorithm/](algorithm/)): Register via `ALGORITHM_REGISTRY`. Must implement `train_step(batch)` and declare `REQUIRED_MODELS`.
- **WeightSync** ([core/weight_sync.py](core/weight_sync.py)): Queue-based sync between training and inference processes.

## Data Types & Conventions
All types in [data/types.py](data/types.py): `Observation`, `RobotState`, `Action`, `Transition`, `Episode`, `Batch`.

**Critical:** Set `Transition.source` ("demo"/"rollout"/"intervention") before writing to DataHub for tracking.

**Action spaces:** `"joint"` (default) | `"cartesian"` | `"delta"` - must align with env config.

**RobotState priority:** Uses `raw_state` if set, else concatenates `joint_pos`, `joint_vel`, `ee_pos`, `ee_quat`, `gripper_pos`.

## Config System
- **YAML config:** [config/train_config.yaml](config/train_config.yaml) with named configs (e.g., `offline_bc`, `offline_td3bc`)
- **Load:** `load_config_from_yaml(path, name)` from [config/config.py](config/config.py)
- **Factory helpers:** `make_bc_config()`, `make_td3bc_config()`, `make_sac_config()`
- **Algorithm-specific params:** Use `algo_kwargs` dict in `AlgorithmConfig` for non-standard params

## Multi-Stage Training
Define stages in `training.stages[]` with different algorithms, sample strategies, and active models:
```yaml
training:
  stages:
    - name: "pretrain_vf"
      algorithm: "vf_regression"
      max_steps: 50000
      active_models: ["vf"]
      sample_strategy: "demo_only"
    - name: "train_policy"
      algorithm: "awr"
      max_steps: 100000
      active_models: ["policy"]
      sample_strategy: "mixed"
      sample_kwargs:
        demo_ratio: 0.3
```
Stage automatically freezes/unfreezes models based on `active_models`.

## Reward Module
Reward functions in `reward/` with registry pattern:

| Type | Class | Use Case |
|------|-------|----------|
| `env` | `EnvReward` | Use environment's raw reward |
| `shaped` | `PotentialShapingReward` | φ(s') - φ(s) shaping |
| `rnd` | `RNDReward` | Intrinsic curiosity reward |
| `composite` | `CompositeReward` | Combine multiple rewards |

```python
from reward import create_reward, CompositeReward, EnvReward, RNDReward

# Simple usage
reward_fn = create_reward("env", scale=1.0, shift=0.0)

# Composite reward (extrinsic + intrinsic)
composite = CompositeReward()
composite.add(EnvReward(), weight=1.0)
composite.add(RNDReward(state_dim=16), weight=0.1)

# Transform batch rewards before training
batch = reward_fn.transform_batch(batch)
```

## Logger Module
Unified logging in `logger/` supporting multiple backends:

```python
from logger import create_logger, MetricsTracker

# Create multi-backend logger
logger = create_logger(
    name="sac_experiment",
    log_dir="./logs",
    use_console=True,
    use_tensorboard=True,
    use_wandb=False,
)

# Log training metrics
logger.log_scalars({"q_loss": 0.5, "policy_loss": 0.3}, step=100, prefix="train")
logger.info("Training started")

# Track episode statistics
tracker = MetricsTracker(window_size=100)
tracker.add_episode_step(reward=1.0, done=False)
tracker.end_episode(success=True)
print(tracker.get_episode_stats())

logger.close()
```

## Adding New Algorithm (Example: DQN)
1. Create `algorithm/dqn.py`:
```python
class DQN(BaseAlgorithm):
    REQUIRED_MODELS = ["q_network", "target_q"]
    
    def train_step(self, batch: Batch) -> Dict[str, float]:
        # Compute TD loss, update q_network, sync to target_q
        return {"td_loss": loss.item()}
```
2. Register in `algorithm/__init__.py`: `ALGORITHM_REGISTRY["dqn"] = DQN`
3. Add to `config/train_config.yaml` under `configs:`
4. Create models in `scripts/train.py::create_model_group()` if needed
5. Run: `python scripts/train.py --config config/train_config.yaml --name my_dqn_config`

## Model Creation Pattern
In `scripts/train.py::create_model_group()`, add models based on algorithm:
```python
if algo_name in ["td3_bc", "sac", "cql"]:
    q1 = QNetwork(state_dim, action_dim, hidden_dims)
    q2 = QNetwork(state_dim, action_dim, hidden_dims)
    model_group.add("q1", q1, frozen=False)
    model_group.add("q2", q2, frozen=False)
    model_group.add("target_q1", copy.deepcopy(q1), frozen=True)
    model_group.add("target_q2", copy.deepcopy(q2), frozen=True)
```

## Workflows
```bash
# Offline BC training
python scripts/train.py --config config/train_config.yaml --name offline_bc

# Offline TD3+BC training
python scripts/train.py --config config/train_config.yaml --name offline_td3bc

# Override config via CLI (not implemented - edit YAML instead)
```

## Critical Patterns & Gotchas
- **Empty buffer handling:** Samplers return empty list if buffer empty - training loop waits with `time.sleep(0.1)`
- **Device handling:** Batch has `.to(device)` method - call before training
- **Optimizer creation:** Each algorithm creates its own optimizer in `__init__` from `model_group.trainable_parameters()`
- **Target network updates:** Must manually implement soft updates in algorithm (see `td3_bc.py`)
- **Gradient clipping:** Standard practice is `torch.nn.utils.clip_grad_norm_(params, 1.0)`
- **Model validation:** Call `self._validate_model_group()` in algorithm `__init__` to check `REQUIRED_MODELS`
- **Checkpoint saving:** Training loop saves via `model_group.save()` at `config.training.save_freq` steps
- **DataHub bool check:** Use `if data_hub is not None:` instead of `if data_hub:` (DataHub.__len__ returns 0 when empty)
