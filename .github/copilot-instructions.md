# VLA-RL AI Coding Guide

## Architecture Overview
Modular RL framework for **Offline/Online/Human-in-the-Loop** robot training. Core pattern: **Registry + Factory** for all extensible components.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        VLA-RL 统一架构                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐               │
│  │ Demo(HDF5)  │     │ Rollout     │     │Intervention │               │
│  │  (只读)     │     │  (FIFO)     │◀────│  (持久化)   │               │
│  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘               │
│         │                   │ ▲                 │                       │
│         └───────────────────┼─┼─────────────────┘                       │
│                             │ │ write                                   │
│                             ▼ │                                         │
│                      ┌──────────────┐                                   │
│                      │   DataHub    │  ← 统一数据接口                    │
│                      └──────┬───────┘                                   │
│                             │ sample(strategy)                          │
│                             ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                     TrainingLoop                                  │  │
│  │  ┌─────────────┐              ┌─────────────────────────────┐    │  │
│  │  │ Algorithm   │─────────────▶│      ModelGroup              │    │  │
│  │  │ (BC/SAC/TD3)│  optimizer   │  ┌──────┬────┬────────────┐ │    │  │
│  │  └─────────────┘   update     │  │policy│ Q  │ target_Q   │ │    │  │
│  │                               │  └──────┴────┴────────────┘ │    │  │
│  └───────────────────────────────┴──────────────┬──────────────┘────┘  │
│                                                 │ push weights          │
│                                                 ▼                       │
│                                          ┌──────────────┐               │
│                                          │ WeightSync   │               │
│                                          │ (Queue/SHM)  │               │
│                                          └──────┬───────┘               │
│                                                 │ pull weights          │
│                                                 ▼                       │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                     InferenceLoop                                 │  │
│  │  ┌─────────────┐     action      ┌─────────────┐                 │  │
│  │  │   Policy    │────────────────▶│     Env     │                 │  │
│  │  │  (推理用)   │◀────────────────│ (真实/Fake) │                 │  │
│  │  └─────────────┘  obs+reward     └─────────────┘                 │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Online RL 数据流详解

### 启动命令
```bash
python scripts/train_online.py --config config/train_config.yaml --name online_sac_dummy --warmup 1000
```

### 数据流转过程 (适合例会汇报)

```
Step 1: 环境交互 (InferenceLoop)
┌────────────────────────────────────────────────────────────────────┐
│ Env.reset() → EnvOutput(obs, robot_state, reward=0, done=False)    │
│                                                                    │
│ Loop:                                                              │
│   Policy.forward(robot_state) → Action                             │
│   Env.step(action) → EnvOutput(obs', robot_state', reward, done)   │
│   Transition = (s, a, r, s', done, source="rollout")               │
│   DataHub.write(transition, source="rollout")                      │
└────────────────────────────────────────────────────────────────────┘

Step 2: 数据存储 (DataHub)
┌────────────────────────────────────────────────────────────────────┐
│ DataHub.write(transition, "rollout")                               │
│   → RolloutBuffer.add_transition(transition)                       │
│   → FIFO 环形缓冲区，容量 100000                                    │
└────────────────────────────────────────────────────────────────────┘

Step 3: 数据采样 (Sampler)
┌────────────────────────────────────────────────────────────────────┐
│ DataHub.sample(batch_size=256, strategy="rollout_only")            │
│   → RolloutOnlyStrategy.sample(buffers, 256)                       │
│   → List[Transition] → Batch(robot_state, action, reward, ...)     │
└────────────────────────────────────────────────────────────────────┘

Step 4: 模型训练 (TrainingLoop)
┌────────────────────────────────────────────────────────────────────┐
│ batch = batch.to(device)  # GPU 传输                               │
│ metrics = algorithm.train_step(batch)                              │
│   → 计算 Q-loss, Policy-loss                                       │
│   → optimizer.step()                                               │
│   → soft_update(target_q, q)  # τ=0.005                            │
└────────────────────────────────────────────────────────────────────┘

Step 5: 权重同步 (WeightSync)
┌────────────────────────────────────────────────────────────────────┐
│ Training: weight_sync.push({"policy": state_dict}, version)        │
│ Inference: weight_sync.pull() → 更新推理用 Policy                   │
│ 方式: Queue(默认) 或 SharedMemory(高频)                             │
└────────────────────────────────────────────────────────────────────┘
```

## Fake Env 实现指南 (DummyEnv)

### 核心接口
```python
class BaseEnv(ABC):
    def __init__(self, config: EnvConfig):
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
    
    @abstractmethod
    def reset(self, task_id=None) -> EnvOutput: ...
    
    @abstractmethod  
    def step(self, action: Action) -> EnvOutput: ...
```

### DummyEnv 实现示例 (见 [env/dummy_env.py](env/dummy_env.py))
```python
class DummyEnv(BaseEnv):
    """任务：让状态接近目标位置"""
    
    def reset(self, task_id=None) -> EnvOutput:
        self._current_state = np.random.randn(self.state_dim) * 0.1
        self._target_state = np.zeros(self.state_dim)  # 目标是原点
        return EnvOutput(
            obs=Observation(images={}, language="reach target"),
            robot_state=RobotState(joint_pos=np.zeros(7), raw_state=self._current_state),
            reward=0.0, done=False, info={"task_id": task_id}
        )
    
    def step(self, action: Action) -> EnvOutput:
        # 状态转移: s' = s + a * scale + noise
        self._current_state += action.data[:self.state_dim] * 0.1 + np.random.randn(...) * 0.01
        
        # 奖励计算
        dist = np.linalg.norm(self._current_state - self._target_state)
        reward = -dist * 0.1
        success = dist < 0.5
        if success: reward = 10.0
        
        done = self._step_count >= max_steps or success
        return EnvOutput(obs, robot_state, reward, done, {"success": success})
```

### 自定义 Reward 函数
```python
# 方式1: 传入 reward_fn
env = DummyEnv(config, reward_fn=lambda s, a, s_next: -np.sum(a**2))

# 方式2: 使用 Reward 模块 (推荐)
from reward import create_reward, CompositeReward, EnvReward

reward_fn = create_reward("env", scale=1.0)  # 直接用环境奖励
# 或组合多个奖励
composite = CompositeReward()
composite.add(EnvReward(), weight=1.0)
composite.add(RNDReward(state_dim=65), weight=0.1)  # 内在好奇心
```

## 模块间数据传输格式

| 传输点 | 数据类型 | 关键字段 |
|--------|----------|----------|
| Env → InferenceLoop | `EnvOutput` | obs, robot_state, reward, done, info |
| InferenceLoop → DataHub | `Transition` | obs, robot_state, action, reward, next_*, done, **source** |
| DataHub → TrainingLoop | `Batch` | robot_state [B,D], action [B,A], reward [B], done [B] |
| TrainingLoop → WeightSync | `Dict[str, state_dict]` | {"policy": {...}} |
| WeightSync → InferenceLoop | `Tuple[Dict, int]` | (state_dict, version) |

## Registry Pattern (添加新组件)

| Component | Base Class | Registry Location |
|-----------|-----------|------------------|
| Algorithm | `BaseAlgorithm` | `algorithm/__init__.py` |
| Policy | `BasePolicy` | `model/__init__.py` |
| Sampler | `BaseSampleStrategy` | `buffer/sample_strategy.py` |
| Environment | `BaseEnv` | `env/__init__.py` |
| Reward | `BaseReward` | `reward/__init__.py` |
| WeightSync | `BaseWeightSync` | `core/weight_sync.py` |

```python
# 添加新算法示例
class MyAlgo(BaseAlgorithm):
    REQUIRED_MODELS = ["policy", "q1", "q2"]
    def train_step(self, batch: Batch) -> Dict[str, float]: ...

ALGORITHM_REGISTRY["my_algo"] = MyAlgo
```

## 关键配置 (train_config.yaml)

```yaml
online_sac_dummy:
  env:
    name: "dummy"
    state_dim: 16
    action_dim: 8
  
  data:
    type: "buffer"
    rollout_capacity: 100000
  
  algorithm:
    name: "sac"
    batch_size: 256
    gamma: 0.99
    tau: 0.005        # target network soft update
    alpha: 0.2        # entropy coefficient
  
  training:
    stages:
      - name: "online_sac"
        max_steps: 100000
        sample_strategy: "rollout_only"
  
  weight_sync:
    method: "queue"   # or "shared_memory"
    sync_freq: 100
```

## Commands
```bash
# Offline BC
python scripts/train.py --config config/train_config.yaml --name offline_bc

# Offline TD3+BC  
python scripts/train.py --config config/train_config.yaml --name offline_td3bc

# Online SAC (with DummyEnv)
python scripts/train_online.py --config config/train_config.yaml --name online_sac_dummy --warmup 1000

# Two-Stage Training (Offline → Online) - Example
python scripts/train_two_stage_example.py
```

## Critical Patterns
- **Transition.source**: 必须设置 `"demo"` / `"rollout"` / `"intervention"`
- **Device handling**: 训练前调用 `batch.to(device)`
- **Target networks**: 需手动实现 soft update (见 [algorithm/sac.py](algorithm/sac.py))
- **Empty buffer**: Samplers 返回 `[]`；训练循环等待 `time.sleep(0.1)`
- **DataHub check**: 用 `if data_hub is not None:` 而非 `if data_hub:`
- **Warmup**: Online RL 需要先收集足够数据再开始训练

## Two-Stage Training (Offline → Online)

### 设计思路
```
Stage 1: Offline TD3+BC (Demo Only)
  └── 用 Demo 数据训练基础 Policy + Q 网络
  
Stage 2: Online SAC (Demo + Rollout Mixed)
  └── 继承 Stage 1 权重，在环境中 finetune
  └── Rollout 初始为空，MixedStrategy 自动退化为 demo_only
  └── 随着交互，Rollout 逐渐增加，最终达到 25% Demo + 75% Rollout
```

### 运行示例
```bash
python scripts/train_two_stage_example.py
```

**预期输出**:
```
Stage 1: Offline TD3+BC (Demo Only)
  Demo: 50 条, Rollout: 0 条
  训练 500 步 → q_loss 降低到 <1.0

Stage 2: Online SAC (Demo + Rollout Mixed)
  Rollout 从 0 逐渐增加到 2000
  采样自动调整: 初期纯 Demo → 后期 25% Demo + 75% Rollout
  最终 avg_reward: 10.0 (成功完成任务)
```

### 关键验证点
- ✅ Rollout 为空时采样不报错（自动退化）
- ✅ 混合采样比例正确 (demo_ratio=0.25)
- ✅ Stage 1 → Stage 2 权重继承成功
- ✅ Online 训练收敛
