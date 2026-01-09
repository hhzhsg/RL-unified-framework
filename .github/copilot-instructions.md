# VLA-RL AI Coding Guide

## Architecture Overview
Training and inference run separately, synchronized via WeightSync queues. Training pulls batches from DataHub buffers and pushes weights; inference pulls weights and executes policy in env.

```
Intervention/Demo/Rollout → Sampler → TrainingLoop → Algorithm → ModelGroup
         ↑                                                          ↓ sync
      write                                                    InferenceLoop ↔ Env
```

## Key Components

- **DataHub** ([buffer/data_hub.py](buffer/data_hub.py)): Unified interface for 3 data sources. Demo=HDF5 read-only, Rollout=FIFO ring buffer, Intervention=with disk persistence.
- **Sampler** ([buffer/sample_strategy.py](buffer/sample_strategy.py)): `demo_only`/`rollout_only`/`mixed` strategies via `STRATEGY_REGISTRY`.
- **ModelGroup** ([model/model_group.py](model/model_group.py)): Central registry with freeze flags. Add frozen pretrained bases + trainable components.
- **TrainingLoop** ([core/training_loop.py](core/training_loop.py)): Multi-stage via `Stage`, auto freeze/unfreeze per stage.
- **Algorithm** ([algorithm/](algorithm/)): Register via `ALGORITHM_REGISTRY`. BC/SAC as examples.
- **WeightSync** ([core/weight_sync.py](core/weight_sync.py)): Queue-based sync between training and inference processes.

## Data Types
All in [data/types.py](data/types.py): `Observation`, `RobotState`, `Action`, `Transition`, `Episode`, `Batch`. Set `Transition.source` before writing to DataHub.

## Config System
- YAML config: [config/train_config.yaml](config/train_config.yaml)
- Load via `load_config_from_yaml(path, name)` from [config/config.py](config/config.py)
- Factory helpers: `make_bc_config`, `make_sac_config`, `make_recap_config`, `make_hil_config`

## Adding New Algorithm (Offline)
1. Create `algorithm/my_algo.py` inheriting `BaseAlgorithm`
2. Implement `train_step(batch) -> Dict[str, float]`
3. Add to `ALGORITHM_REGISTRY` in `algorithm/__init__.py`
4. Add config in `config/train_config.yaml`
5. Run: `python scripts/train.py --config config/train_config.yaml --name my_config`

## Adding New Environment
1. Create `env/my_env.py` inheriting `BaseEnv`
2. Implement `reset() -> EnvOutput`, `step(action) -> EnvOutput`
3. Add to `ENV_REGISTRY` in `env/__init__.py`

## Workflows
```bash
# Offline BC training
python scripts/train.py --config config/train_config.yaml --name offline_bc

# Override parameters
python scripts/train.py --config config/train_config.yaml --name offline_bc --steps 10000
```

## Gotchas
- Align `Action.space` with env (default "joint")
- Guard against empty buffers in custom sample strategies
- Weight sync frequency: tune `config.weight_sync.sync_freq` to avoid overhead
- Multi-stage: use `active_models` to control which models train per stage
