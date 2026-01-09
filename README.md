# VLA-RL Framework

**模块化强化学习框架** — 专为机器人学习设计，支持 **Offline / Online / Human-in-the-Loop** 三种训练模式。

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✨ 核心特性

- 🧩 **积木式架构**：注册表模式支持插件化扩展，无需修改框架代码
- 🎯 **单一职责**：每个模块专注一件事，易于维护和测试
- 📝 **配置驱动**：YAML 配置文件组合不同模块，10 分钟启动新实验
- 🔄 **多阶段训练**：原生支持 RECAP/AWR 等多阶段算法
- 🤖 **机器人友好**：统一管理 Demo/Rollout/Intervention 三种数据源
- ⚡ **训练/推理分离**：异步架构，互不阻塞

---

## 📦 版本历史

### v0.2.0 (2026-01-09) - 当前版本
- ✅ 完整的多阶段训练支持
- ✅ TD3+BC / SAC 离线算法实现
- ✅ ModelGroup 冻结/解冻控制
- ✅ 配置驱动的训练流程
- ✅ 完善的类型提示和文档

### v0.1.0 (2025-12) - 初始版本
- ✅ 基础框架搭建
- ✅ BC 算法实现
- ✅ DataHub 数据管理
- ✅ HDF5 数据加载

```
                    ┌─────────────┐
                    │ Intervention│───┐
                    └─────────────┘   │
                    ┌─────────────┐   │     ┌─────────┐      ┌──────────────┐
                    │    Demo     │───┼────▶│ Sampler │─────▶│Training Loop │
                    └─────────────┘   │     └─────────┘      │  ┌─────────┐ │
                    ┌─────────────┐   │                      │  │Algorithm│ │
                    │   Rollout   │───┘                      │  └─────────┘ │
                    └─────────────┘                          └──────┬───────┘
                          ▲                                         │
                          │ write                                   │ sync
                          │                                         ▼
┌─────────┐  obs/state   ┌──────────────┐                   ┌──────────────┐
│   Env   │◀────────────▶│Inference Loop│◀─────────────────▶│ ModelGroup   │
└─────────┘    action    └──────────────┘                   │ ┌──────┬───┐ │
                                                            │ │policy│ VF│ │
                                                            │ └──────┴───┘ │
                                                            └──────────────┘
```

## 目录结构

```
vla_rl/
├── config/                 # 🔧 配置系统
│   ├── config.py           #    Config/StageConfig 等 dataclass 定义
│   ├── train_config.yaml   #    YAML 训练配置文件
│   └── __init__.py
│
├── data/                   # 📦 数据类型定义
│   ├── types.py            #    Observation, RobotState, Action, Transition, Episode, Batch
│   └── __init__.py
│
├── buffer/                 # 💾 数据管理 (对应架构图左上)
│   ├── data_hub.py         #    DataHub: 统一数据接口
│   ├── hdf5_buffer.py      #    HDF5DemoBuffer: Demo 数据加载
│   ├── rollout_buffer.py   #    RolloutBuffer: 在线数据存储
│   ├── intervention_buffer.py  # InterventionBuffer: 人工干预数据
│   ├── sample_strategy.py  #    Sampler: 采样策略 (demo_only/mixed/...)
│   └── __init__.py
│
├── model/                  # 🧠 模型定义 (对应架构图右下 ModelGroup)
│   ├── model_group.py      #    ModelGroup: 模型注册/冻结管理
│   ├── base_policy.py      #    BasePolicy: 策略基类
│   ├── mlp_policy.py       #    MLPPolicy/MLPGaussianPolicy
│   ├── composite_policy.py #    ResidualPolicy/EnsemblePolicy
│   └── __init__.py
│
├── algorithm/              # ⚙️ 训练算法 (对应架构图 Algorithm)
│   ├── base_algorithm.py   #    BaseAlgorithm: 算法基类
│   ├── bc.py               #    BC: Behavior Cloning
│   ├── sac.py              #    SAC: Soft Actor-Critic
│   └── __init__.py         #    ALGORITHM_REGISTRY
│
├── core/                   # 🔄 核心循环 (对应架构图 Training/Inference Loop)
│   ├── training_loop.py    #    TrainingLoop: 训练主循环
│   ├── inference_loop.py   #    InferenceLoop: 推理主循环
│   ├── weight_sync.py      #    WeightSync: 权重同步 (sync 箭头)
│   ├── stage.py            #    Stage: 多阶段训练控制
│   └── __init__.py
│
├── env/                    # 🌍 环境接口 (对应架构图 Env)
│   ├── base_env.py         #    BaseEnv: 环境基类
│   ├── dummy_env.py        #    DummyEnv: 测试用虚拟环境
│   └── __init__.py         #    ENV_REGISTRY
│
├── scripts/                # 🚀 运行脚本
│   └── train.py            #    统一训练入口
│
└── checkpoints/            # 💾 模型存储
```

---

## 🚀 快速开始

### 安装

#### 1. 克隆仓库
```bash
git clone https://github.com/hhzhsg/RL-unified-framework.git
cd RL-unified-framework
```

#### 2. 创建 Python 环境（推荐）
```bash
# 使用 conda
conda create -n vla_rl python=3.10
conda activate vla_rl

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

#### 3. 安装依赖
```bash
pip install -r requirements.txt
```

**依赖包说明**：
- `torch>=2.0.0` - 深度学习框架
- `numpy>=1.24.0` - 数值计算
- `h5py>=3.8.0` - HDF5 数据格式支持
- `pyyaml>=6.0` - YAML 配置文件解析

#### 4. 验证安装
```bash
python -c "import torch; import numpy; import h5py; import yaml; print('✅ 所有依赖安装成功')"
```

---

### 使用示例

#### 方式 1：使用配置文件训练（推荐）

```bash
# 离线 BC 训练
python scripts/train.py --config config/train_config.yaml --name offline_bc

# 离线 TD3+BC 训练
python scripts/train.py --config config/train_config.yaml --name offline_td3bc
```

#### 方式 2：编程式训练

```python
from config import make_bc_config, load_config_from_yaml
from model import ModelGroup, MLPPolicy
from buffer import DataHub
from core import TrainingLoop
from scripts.train import create_model_group

# 1. 加载配置
config = load_config_from_yaml("config/train_config.yaml", "offline_bc")

# 2. 创建数据中心
data_hub = DataHub(
    demo_paths=config.data.demo_paths,
    load_images=False
)

# 3. 创建模型组
model_group = create_model_group(config)

# 4. 启动训练
trainer = TrainingLoop(
    model_group=model_group,
    data_hub=data_hub,
    config=config.training,
    algo_config=config.algorithm,
    device=config.device
)
trainer.run()
```

#### 方式 3：10 行代码开始实验

```python
from config import make_bc_config
from scripts.train import setup_logging, create_model_group, create_data_hub
from core import TrainingLoop

config = make_bc_config()  # 使用预设配置
logger = setup_logging(config.exp_name)
data_hub = create_data_hub(config)
model_group = create_model_group(config)
trainer = TrainingLoop(model_group, data_hub, config.training, config.algorithm, config.device)
trainer.run()
```

---

## 设计理念

### 积木式架构

框架采用 **基类 + 继承 + 注册表** 的设计模式，各模块可像积木一样自由组合：

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            积木式组合                                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Env 积木           Buffer 积木         Algorithm 积木      Policy 积木        │
│   ┌─────────┐       ┌─────────┐         ┌─────────┐        ┌─────────┐        │
│   │ BaseEnv │       │DataHub  │         │BaseAlgo │        │BasePolicy│        │
│   └────┬────┘       └────┬────┘         └────┬────┘        └────┬────┘        │
│        │                 │                   │                  │              │
│   ┌────┼────┐       ┌────┼────┐         ┌────┼────┐        ┌────┼────┐        │
│   ▼    ▼    ▼       ▼    ▼    ▼         ▼    ▼    ▼        ▼    ▼    ▼        │
│ Real  Sim  Dummy  Demo Rollout Intv    BC  SAC  CQL      MLP  VLA Residual   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

组合示例:
  Offline BC  = DummyEnv   + DemoBuffer      + BC     + MLPPolicy
  Online SAC  = RealEnv    + RolloutBuffer   + SAC    + GaussianPolicy  
  HIL         = RealEnv    + Mixed(3 Buffer) + SAC    + ResidualPolicy
  RECAP (π₀*) = SimEnv     + DemoBuffer      + [VF→AWR] + VLAPolicy
```

### 基类定义核心接口

每个模块有统一的基类，定义必须实现的接口：

```python
# 环境基类
class BaseEnv(ABC):
    @abstractmethod
    def reset(self) -> EnvOutput: ...
    @abstractmethod  
    def step(self, action) -> EnvOutput: ...

# 算法基类
class BaseAlgorithm(ABC):
    @abstractmethod
    def train_step(self, batch) -> Dict[str, float]: ...

# 策略基类
class BasePolicy(ABC):
    @abstractmethod
    def forward(self, obs, state) -> Tensor: ...
    @abstractmethod
    def act(self, obs, state) -> Action: ...

# Buffer 基类
class BaseBuffer(ABC):
    @abstractmethod
    def add(self, data): ...
    @abstractmethod
    def sample(self, batch_size) -> Batch: ...
```

### 注册表实现即插即用

新增实现后注册到对应表，即可在配置中使用：

```python
# 1. 继承基类实现
class MyCustomEnv(BaseEnv):
    def reset(self): ...
    def step(self, action): ...

# 2. 注册
ENV_REGISTRY["my_env"] = MyCustomEnv

# 3. 配置中使用
# env:
#   name: "my_env"
```

**注册表映射关系：**

| 模块 | 基类 | 注册表 | 配置字段 |
|------|------|--------|---------|
| 环境 | `BaseEnv` | `ENV_REGISTRY` | `env.name` |
| 算法 | `BaseAlgorithm` | `ALGORITHM_REGISTRY` | `algorithm.name` |
| 策略 | `BasePolicy` | `POLICY_REGISTRY` | `model.policy_type` |
| 采样 | `BaseSampleStrategy` | `STRATEGY_REGISTRY` | `sample_strategy` |
| 同步 | `BaseWeightSync` | `WEIGHT_SYNC_REGISTRY` | `weight_sync.method` |

### 扩展任意模块的统一流程

```
1. 找到基类 → 2. 继承实现 → 3. 注册 → 4. 配置使用
```

**示例：新增 PPO 算法**

```python
# algorithm/ppo.py
class PPO(BaseAlgorithm):
    def train_step(self, batch):
        # ... PPO 实现
        return {"policy_loss": ..., "value_loss": ...}

# algorithm/__init__.py  
ALGORITHM_REGISTRY["ppo"] = PPO

# config/train_config.yaml
algorithm:
  name: "ppo"  # 直接使用
```

---

## 快速开始：配置驱动训练

### 1. 编辑配置文件

```yaml
# config/train_config.yaml
configs:
  my_bc_exp:
    exp_name: "my_bc_experiment"
    device: "cuda"
    
    env:
      state_dim: 65
      action_dim: 37
    
    data:
      type: "hdf5"
      demo_paths:
        - "/path/to/demos/**/*.hdf5"
      load_images: false
    
    model:
      policy_type: "mlp"
      hidden_dims: [256, 256, 128]
    
    algorithm:
      name: "bc"
      lr: 3.0e-4
      batch_size: 64
    
    training:
      stages:
        - name: "bc_train"
          algorithm: "bc"
          max_steps: 10000
          active_models: ["policy"]
          sample_strategy: "demo_only"
      checkpoint_dir: "./checkpoints/my_bc"
```

### 2. 运行训练

```bash
python scripts/train.py --config config/train_config.yaml --name my_bc_exp
```

### 3. 查看结果

```
checkpoints/my_bc/
└── final_policy.pt    # 训练好的模型
```

---

## 完整场景：从零构建新训练流程

假设你要实现 **DQN (Deep Q-Network) + 优先级经验回放 (PER)** 的离线训练。

### 场景需求分析

```
目标: 使用 Q-learning 训练策略
数据: 专家演示 (HDF5)
算法: DQN + Double DQN + PER
策略: Q 网络 (输出每个动作的 Q 值)
采样: 优先级采样 (TD-error 作为优先级)
```

### 实现步骤

#### 1. 实现 Q 网络策略

```python
# model/q_network.py
class QNetwork(BasePolicy):
    def __init__(self, state_dim, action_dim, hidden_dims):
        # ... 构建 MLP 网络
    
    def forward(self, obs, robot_state):
        return self.network(robot_state)  # → (B, action_dim) Q值
    
    def act(self, obs, robot_state, deterministic=True):
        q_values = self.forward(...)
        action_idx = q_values.argmax() if deterministic else epsilon_greedy(...)
        return Action(data=action_idx, space="discrete")

# model/__init__.py
POLICY_REGISTRY["q_network"] = QNetwork
```

#### 2. 实现 DQN 算法

```python
# algorithm/dqn.py
class DQN(BaseAlgorithm):
    def __init__(self, model_group, config):
        self.q_network = model_group.get("q_network")
        self.target_q = model_group.get("target_q")
        # ... 初始化优化器
    
    def train_step(self, batch):
        # 计算当前 Q 值
        q_values = self.q_network.forward(...)
        
        # 计算目标 Q 值 (Double DQN)
        target_q = reward + gamma * target_q_network(next_state)
        
        # TD Loss
        loss = mse_loss(q_values, target_q)
        loss.backward()
        optimizer.step()
        
        # 软更新 target 网络
        self._soft_update_target()
        
        return {"q_loss": loss.item()}

# algorithm/__init__.py
ALGORITHM_REGISTRY["dqn"] = DQN
```

#### 3. 实现优先级采样策略

```python
# buffer/prioritized_strategy.py
class PrioritizedReplayStrategy(BaseSampleStrategy):
    def sample(self, buffers, batch_size):
        # 计算采样概率: P(i) = priority[i]^α / Σ priority^α
        probs = priorities ** self.alpha / sum(priorities ** self.alpha)
        indices = np.random.choice(len(buffer), size=batch_size, p=probs)
        return [buffer[i] for i in indices]
    
    def update_priorities(self, indices, td_errors):
        self.priorities[indices] = abs(td_errors) + ε

# buffer/sample_strategy.py
STRATEGY_REGISTRY["prioritized"] = PrioritizedReplayStrategy
```

#### 4. 扩展 train.py 创建模型

```python
# scripts/train.py
def create_model_group(config):
    if config.algorithm.name == "dqn":
        q_network = QNetwork(state_dim, action_dim, hidden_dims)
        target_q = copy.deepcopy(q_network)
        
        model_group.add("q_network", q_network, frozen=False)
        model_group.add("target_q", target_q, frozen=True)
```

#### 5. 配置文件

```yaml
# config/train_config.yaml
configs:
  dqn_offline:
    model:
      policy_type: "q_network"
    algorithm:
      name: "dqn"
      tau: 0.005  # 软更新系数
    training:
      stages:
        - algorithm: "dqn"
          active_models: ["q_network"]
          sample_strategy: "prioritized"
          sample_kwargs: {alpha: 0.6, beta: 0.4}
```

#### 6. 运行

```bash
python scripts/train.py --config config/train_config.yaml --name dqn_offline
```

### 训练数据流

```
HDF5 → HDF5DemoBuffer → PrioritizedSampler → Batch
         (lazy load)      (TD-error权重)      ↓
                                        Q-Network.forward()
                                              ↓
                                        TD Loss 计算
                                              ↓
                                        反向传播 + 梯度更新
                                              ↓
                                        软更新 Target Q
                                              ↓
                                        更新采样优先级
```

### 扩展优势总结

通过这个完整案例可以看到：

1. **无需修改核心代码**：所有新增功能都通过继承基类实现
2. **注册即可用**：新模块注册后立即可在配置中引用
3. **配置驱动**：通过 YAML 配置组合不同模块
4. **复用现有基础设施**：
   - DataHub 处理数据加载
   - TrainingLoop 管理训练循环
   - ModelGroup 管理模型注册/冻结

**框架的价值在于**：将重复的"脚手架代码"抽象成可复用的基础设施，让开发者专注于算法本身的实现。

---

## 开发指南：添加新的 Offline 算法

### CQL (Conservative Q-Learning) 示例

#### 1. 实现算法类

```python
# algorithm/cql.py
class CQL(BaseAlgorithm):
    def __init__(self, model_group, config):
        self.q1 = model_group.get("q1")
        self.cql_alpha = config.cql_alpha
    
    def train_step(self, batch):
        # 标准 Q-loss
        q_loss = compute_td_loss(...)
        # Conservative 惩罚项
        cql_loss = (logsumexp(Q_values) - Q(s, a_real)).mean()
        total_loss = q_loss + self.cql_alpha * cql_loss
        # backward + optimize ...
        return {"q_loss": q_loss.item(), "cql_loss": cql_loss.item()}

# algorithm/__init__.py
ALGORITHM_REGISTRY["cql"] = CQL
```

#### 2. 添加配置

```yaml
# config/train_config.yaml
configs:
  cql_offline:
    algorithm:
      name: "cql"
      cql_alpha: 5.0  # Conservative 惩罚系数
    training:
      stages:
        - algorithm: "cql"
          max_steps: 50000
```

#### 3. 运行

```bash
python scripts/train.py --config config/train_config.yaml --name cql_offline
```

---

## 核心概念

### DataHub：三源数据管理

```python
from buffer import DataHub

data_hub = DataHub(
    demo_paths=["demos/*.hdf5"],     # Demo: 专家演示 (只读)
    rollout_capacity=100000,          # Rollout: 策略采集 (FIFO)
    intervention_capacity=50000,      # Intervention: 人工干预
)

# 写入数据
data_hub.write(episode, source="rollout")       # 在线采集
data_hub.write(transition, source="intervention") # 人工干预

# 采样数据 (通过 Sampler)
batch = data_hub.sample(batch_size=64, strategy="demo_only")
batch = data_hub.sample(batch_size=64, strategy="mixed", demo_ratio=0.5)
```

### ModelGroup：模型注册与冻结

```python
from model import ModelGroup, MLPPolicy

model_group = ModelGroup()

# 添加模型
model_group.add("policy", MLPPolicy(...), frozen=False)
model_group.add("base_policy", pretrained_vla, frozen=True)  # 冻结

# 阶段性冻结/解冻
model_group.freeze("policy")
model_group.unfreeze("policy")

# 获取可训练参数
params = model_group.trainable_parameters(["policy"])
```

### Stage：多阶段训练

```yaml
training:
  stages:
    - name: "train_vf"        # 阶段1: 训练价值函数
      algorithm: "vf_regression"
      max_steps: 50000
      active_models: ["vf"]
      sample_strategy: "demo_only"
      
    - name: "train_policy"    # 阶段2: 训练策略
      algorithm: "awr"
      max_steps: 50000
      active_models: ["policy"]
      sample_strategy: "demo_only"
```

### WeightSync：训练/推理分离

```python
from core import TrainingLoop, InferenceLoop, create_weight_sync

# 创建同步器
weight_sync = create_weight_sync("queue")

# 训练进程
trainer = TrainingLoop(..., weight_sync=weight_sync)

# 推理进程 (独立进程)
inference = InferenceLoop(..., weight_sync=weight_sync)

# 训练时自动 push 权重，推理时自动 pull
```

---

## 采样策略

| 策略 | 说明 | 使用场景 |
|-----|------|---------|
| `demo_only` | 仅从 Demo 采样 | Offline BC/CQL |
| `rollout_only` | 仅从 Rollout 采样 | Online RL |
| `mixed` | 混合采样 (可配比例) | HIL/Fine-tuning |

```python
# 自定义采样策略
from buffer import BaseSampleStrategy, STRATEGY_REGISTRY

class PrioritizedStrategy(BaseSampleStrategy):
    def sample(self, buffers, batch_size):
        # 实现优先级采样
        ...

STRATEGY_REGISTRY["prioritized"] = PrioritizedStrategy
```

---

## 注册表模式

框架使用注册表模式支持扩展：

| 注册表 | 位置 | 用途 |
|-------|------|-----|
| `ALGORITHM_REGISTRY` | `algorithm/__init__.py` | 注册新算法 |
| `POLICY_REGISTRY` | `model/__init__.py` | 注册新策略网络 |
| `ENV_REGISTRY` | `env/__init__.py` | 注册新环境 |
| `STRATEGY_REGISTRY` | `buffer/sample_strategy.py` | 注册采样策略 |
| `WEIGHT_SYNC_REGISTRY` | `core/weight_sync.py` | 注册同步方式 |

---

## 📚 文档

> 💡 **不知道看哪个文档？** 参考 [文档导航指南](DOCS_GUIDE.md)

- **[QUICKREF.md](QUICKREF.md)** - 5 分钟速查表 ⚡
  - 常用配置模板
  - API 速查
  - 问题排查

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - 框架设计哲学与架构详解 📖
  - 为什么这个设计是优雅的？
  - 设计模式详解
  - 模块化设计分析
  - 扩展性分析

- **[docs/TWO_STAGE_TRAINING.md](docs/TWO_STAGE_TRAINING.md)** - 两阶段训练指南 🎯
  - Offline → Online 训练流程
  - 数据流架构详解
  - 使用示例和常见问题
  - 相关工作对比

- **[CHANGELOG.md](CHANGELOG.md)** - 版本更新历史 📝
  - 版本特性
  - 升级指南

- **[.github/copilot-instructions.md](.github/copilot-instructions.md)** - AI 编码助手指南 🤖
  - 快速上手指南
  - 注册表模式使用
  - 添加新算法示例

---

## 🤝 贡献

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

---

## 📄 License

本项目基于 MIT 协议开源 - 查看 [LICENSE](LICENSE) 文件了解详情。

---

## 🙏 致谢

- 框架设计参考了 Stable-Baselines3, CleanRL, RLlib 等优秀项目
- 感谢所有贡献者的支持

---

## 📧 联系方式

- **GitHub Issues**: [提交问题](https://github.com/hhzhsg/RL-unified-framework/issues)
- **作者**: hhzhsg

---

**⭐ 如果这个项目对您有帮助，请给个 Star！**
