# RL-unified-framework

Vision-Language-Action 强化学习框架

## 特性

- **模块自由组合**: Env / Buffer / Policy / Algorithm 任意搭配
- **推理/训练分离**: 独立进程，通过权重同步通信
- **多场景支持**: Offline，Online
- **最小修改成本**: 通过注册表机制，算法可随意添加替换

## 目录结构

```
RL-unified-framework/
├── policy/                # 策略模块
│   ├── base.py           # BasePolicy 基类
│   ├── mlp.py            # MLPPolicy, MLPGaussianPolicy
│   └── composite.py      # ResidualPolicy, EnsemblePolicy
│
├── network/               # 网络模块
│   ├── base.py           # BaseNetwork 基类
│   ├── mlp.py            # 通用 MLP 网络
│   └── q_network.py      # QNetwork, VNetwork
│
├── algorithm/             # 算法模块
│   ├── base.py           # BaseAlgorithm 基类
│   ├── bc.py             # Behavior Cloning
│   ├── sac.py            # Soft Actor-Critic
│   └── td3_bc.py         # TD3 + BC
│
├── buffer/                # 缓冲模块
│   ├── base.py           # BaseBuffer 基类
│   ├── replay.py         # ReplayBuffer
│   └── hdf5.py           # HDF5DemoBuffer (lazy loading)
│
├── env/                   # 环境模块
│   ├── base.py           # BaseEnv 基类
│   └── dummy.py          # 测试环境
│
├── data/                  # 数据模块
│   ├── types.py          # Observation, Action, Transition, Batch...
│   ├── hub.py            # DataHub 数据中心
│   ├── sampler.py        # 采样策略
│   └── transforms/       # 数据转换
│       ├── base.py       # Compose, Identity
│       ├── image.py      # ResizeImage, NormalizeImage
│       └── action.py     # NormalizeAction, DeltaAction
│
├── core/                  # 核心模块
│   ├── model_group.py    # ModelGroup 模型组管理
│   ├── training_loop.py  # 训练循环
│   ├── inference_loop.py # 推理循环
│   └── weight_sync.py    # 权重同步
│
├── robot/                 # 机器人适配器
│   ├── base.py           # BaseRobotAdapter
│   └── binocular.py      # 双目机器人适配器
│
├── config/                # 配置模块
│   ├── base.py           # 配置 dataclass
│   └── loader.py         # YAML 加载器
│
├── utils/                 # 工具模块
│   ├── logger.py         # 日志工具
│   └── io.py             # 文件 IO
│
├── scripts/               # 训练脚本
│   ├── train_offline.py  # 离线训练
│   └── train_online.py   # 在线训练
│
├── configs/               # 配置文件
│   └── example.yaml      # 示例配置
│
└── registry.py            # 统一注册表
```

## 快速开始

### 安装依赖

```bash
pip install torch numpy pyyaml h5py
```

### 离线训练 (BC)

```bash
# 使用 dummy 数据测试
python scripts/train_offline.py --name bc_test --steps 1000

# 使用 HDF5 演示数据
python scripts/train_offline.py \
    --demo_paths /path/to/demos/*.hdf5 \
    --state_dim 15 \
    --action_dim 15 \
    --steps 10000
```

### 在线训练 (SAC)

```bash
python scripts/train_online.py --name sac_test --steps 5000
```

### 使用配置文件

```bash
python scripts/train_offline.py \
    --config configs/example.yaml \
    --config_name bc_mlp
```

## 核心概念

### 数据类型

```python
from data import Observation, RobotState, Action, Transition, Batch

# 观测
obs = Observation(
    images={"cam_high": image_array},
    language="pick up the red block"
)

# 机器人状态
state = RobotState(
    joint_pos=np.zeros(7),
    gripper=0.0,
)

# 动作
action = Action(data=np.zeros(7), space="joint")

# 单步数据
transition = Transition(obs, state, action, reward, next_obs, next_state, done)

# 批量数据
batch = Batch.from_transitions([t1, t2, t3])
```

### 策略

```python
from policy import MLPPolicy, MLPGaussianPolicy

# 确定性策略 (BC, TD3+BC)
policy = MLPPolicy(state_dim=16, action_dim=7, hidden_dims=[256, 256])
action = policy.act(obs, state, deterministic=True)

# 随机策略 (SAC)
policy = MLPGaussianPolicy(state_dim=16, action_dim=7)
action, log_prob = policy.sample(state_tensor)
```

### 算法

```python
from algorithm import BC, SAC, TD3BC
from core import ModelGroup

# 构建模型组
group = ModelGroup()
group.add("policy", policy)
group.add("q1", q_network)

# 创建算法
algo = BC(group, config)
algo = SAC(group, config)

# 训练
metrics = algo.train_step(batch)
```

### 训练循环

```python
from core import TrainingLoop, InferenceLoop
from data import DataHub

# 数据中心
hub = DataHub()
hub.register_buffer("demo", demo_buffer)

# 离线训练
loop = TrainingLoop(algorithm, hub, config)
loop.train(num_steps=10000, sample_strategy="demo_only")

# 在线训练 (推理/训练分离)
weight_sync = create_weight_sync("shared_memory")

# 训练端
train_loop = TrainingLoop(algo, hub, config, weight_sync=weight_sync)

# 推理端
infer_loop = InferenceLoop(policy, env, config, data_hub=hub, weight_sync=weight_sync)
```

## 扩展指南

### 添加新算法

```python
from algorithm import BaseAlgorithm

class MyAlgorithm(BaseAlgorithm):
    REQUIRED_MODELS = ["policy", "value"]
    
    def train_step(self, batch):
        # 实现训练逻辑
        loss = ...
        return {"loss": loss}
```

### 添加新策略

```python
from policy import BasePolicy

class MyPolicy(BasePolicy):
    def forward(self, obs, state):
        # 实现前向传播
        return action_tensor
    
    def act(self, obs, state, deterministic=True):
        # 实现推理
        return Action(data=action_array)
```

### 添加机器人适配器

```python
from robot import BaseRobotAdapter, RobotSpec

MY_SPEC = RobotSpec(
    name="my_robot",
    state_dim=14,
    action_dim=7,
    camera_keys=["cam1", "cam2"],
    state_keys=["joint_pos", "gripper"],
    action_keys=["joint_pos", "gripper"],
)

class MyRobotAdapter(BaseRobotAdapter):
    def preprocess(self, raw_data):
        # HDF5 数据转换
        return {"state": ..., "action": ..., "images": ...}
    
    def postprocess(self, model_output):
        # 模型输出转机器人动作
        return robot_action
```

## 验证测试

```bash
# 运行所有测试
python -c "
import sys
sys.path.insert(0, '.')

# 测试数据类型
from data import Observation, RobotState, Action, Transition, Batch
import numpy as np

state = RobotState(raw_state=np.random.randn(16).astype(np.float32))
action = Action(data=np.random.randn(7).astype(np.float32))
print('✓ Data types OK')

# 测试策略
from policy import MLPPolicy, MLPGaussianPolicy
policy = MLPPolicy(16, 7)
print('✓ Policy OK')

# 测试算法
from algorithm import BC
from core import ModelGroup
group = ModelGroup()
group.add('policy', policy)
algo = BC(group)
print('✓ Algorithm OK')

# 测试 Buffer
from buffer import ReplayBuffer
buf = ReplayBuffer(1000)
print('✓ Buffer OK')

# 测试环境
from env import DummyEnv
from config import EnvConfig
env = DummyEnv(EnvConfig())
out = env.reset()
print('✓ Env OK')

print('\\n所有测试通过!')
"
```

## License

MIT
