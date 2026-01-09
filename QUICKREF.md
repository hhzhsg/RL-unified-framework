# VLA-RL 快速参考指南

> 5 分钟速查表，快速查找常用模式和 API

---

## 🎯 30 秒快速开始

```bash
# 克隆 & 安装
git clone https://github.com/hhzhsg/RL-unified-framework.git
cd RL-unified-framework
pip install -r requirements.txt

# 运行训练
python scripts/train.py --config config/train_config.yaml --name offline_bc
```

---

## 📋 目录结构速查

```
核心模块路径速查：
├── algorithm/          # 添加新算法在这里
├── model/              # 添加新策略网络在这里
├── buffer/             # 添加新数据源在这里
├── env/                # 添加新环境在这里
├── core/               # 训练/推理循环（一般不改）
├── config/             # 配置文件和类型定义
└── scripts/train.py    # 训练入口
```

---

## 🔧 常用配置模板

### Offline BC
```yaml
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
```

### Offline TD3+BC
```yaml
algorithm:
  name: "td3_bc"
  lr: 3.0e-4
  gamma: 0.99
  tau: 0.005
  bc_alpha: 2.5        # BC 正则强度
  policy_noise: 0.2
  noise_clip: 0.5
  policy_freq: 2

training:
  stages:
    - algorithm: "td3_bc"
      max_steps: 50000
      active_models: ["policy", "q1", "q2"]
```

### 多阶段训练 (RECAP Style)
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
      sample_strategy: "demo_only"
```

---

## 🧩 添加新算法速查

### 1️⃣ 创建算法文件
```python
# algorithm/my_algo.py
from .base_algorithm import BaseAlgorithm
from data import Batch

class MyAlgo(BaseAlgorithm):
    REQUIRED_MODELS = ["policy"]  # 声明依赖
    
    def train_step(self, batch: Batch) -> Dict[str, float]:
        # 你的训练逻辑
        loss = ...
        return {"loss": loss.item()}
```

### 2️⃣ 注册算法
```python
# algorithm/__init__.py
from .my_algo import MyAlgo
ALGORITHM_REGISTRY["my_algo"] = MyAlgo
```

### 3️⃣ 添加配置
```yaml
# config/train_config.yaml
configs:
  my_experiment:
    algorithm:
      name: "my_algo"
      lr: 1e-4
```

### 4️⃣ 运行
```bash
python scripts/train.py --config config/train_config.yaml --name my_experiment
```

---

## 🎨 ModelGroup 常用操作

```python
from model import ModelGroup, MLPPolicy

# 创建模型组
model_group = ModelGroup()

# 添加模型
model_group.add("policy", policy_net, frozen=False)
model_group.add("pretrained_vla", vla_net, frozen=True)

# 获取模型
policy = model_group.get("policy")
policy = model_group["policy"]  # 等价写法

# 冻结/解冻
model_group.freeze("policy")
model_group.unfreeze("policy")

# 检查状态
if model_group.is_frozen("policy"):
    print("Policy is frozen")

# 获取可训练参数
params = model_group.trainable_parameters(["policy"])
optimizer = torch.optim.Adam(params, lr=1e-4)

# 保存/加载
model_group.save("checkpoint.pt")
model_group.load("checkpoint.pt")

# 迁移设备
model_group.to("cuda")
```

---

## 💾 DataHub 常用操作

```python
from buffer import DataHub

# 创建数据中心
data_hub = DataHub(
    demo_paths=["demos/*.hdf5"],
    rollout_capacity=100000,
    intervention_capacity=50000
)

# 写入数据
data_hub.write(episode, source="rollout")
data_hub.write(transition, source="intervention")

# 采样 - Demo Only
batch = data_hub.sample(
    batch_size=64,
    strategy="demo_only"
)

# 采样 - Mixed
batch = data_hub.sample(
    batch_size=64,
    strategy="mixed",
    demo_ratio=0.5,
    intervention_ratio=0.2
)

# 检查数据量
print(f"Demo: {len(data_hub.demo_buffer)}")
print(f"Rollout: {len(data_hub.rollout_buffer)}")
```

---

## 📊 数据类型速查

### Transition
```python
from data import Transition

t = Transition(
    obs=obs,                    # Observation
    robot_state=robot_state,    # RobotState
    action=action,              # Action
    reward=1.0,                 # float
    next_obs=next_obs,          # Observation
    next_robot_state=next_rs,   # RobotState
    done=False,                 # bool
    source="demo"               # "demo" | "rollout" | "intervention"
)
```

### Batch
```python
batch = Batch(
    obs=obs_dict,          # Dict[str, Tensor]
    robot_state=states,    # Tensor (B, state_dim)
    action=actions,        # Tensor (B, action_dim)
    reward=rewards,        # Tensor (B,)
    next_obs=next_obs_dict,
    next_robot_state=next_states,
    done=dones
)

# 迁移设备
batch = batch.to("cuda")
```

---

## 🎭 Policy 接口速查

```python
from model import BasePolicy

class MyPolicy(BasePolicy):
    def forward(self, obs, robot_state):
        """训练时使用（支持批量）"""
        return self.network(robot_state)  # (B, action_dim)
    
    def act(self, obs, robot_state, deterministic=True):
        """推理时使用（单个样本）"""
        action_tensor = self.forward(...)
        return Action(data=action_tensor.cpu().numpy())
```

---

## 🔍 注册表速查

| 注册表 | 位置 | 用途 |
|-------|------|-----|
| `ALGORITHM_REGISTRY` | `algorithm/__init__.py` | 算法注册 |
| `POLICY_REGISTRY` | `model/__init__.py` | 策略网络注册 |
| `ENV_REGISTRY` | `env/__init__.py` | 环境注册 |
| `STRATEGY_REGISTRY` | `buffer/sample_strategy.py` | 采样策略注册 |

```python
# 查看已注册的算法
from algorithm import ALGORITHM_REGISTRY
print(ALGORITHM_REGISTRY.keys())
# dict_keys(['bc', 'sac', 'td3_bc'])

# 动态创建
from algorithm import create_algorithm
algo = create_algorithm("bc", model_group, config)
```

---

## 🚦 训练流程速查

```python
from core import TrainingLoop

trainer = TrainingLoop(
    model_group=model_group,
    data_hub=data_hub,
    config=training_config,
    algo_config=algorithm_config,
    device="cuda"
)

# 运行训练
trainer.run()

# 带回调的训练
def my_callback(step, metrics):
    print(f"Step {step}: {metrics}")

trainer.run(callback=my_callback)
```

---

## 🐛 常见问题速查

### Q1: 模型不训练 / loss 不下降
```python
# 检查模型是否冻结
print(model_group.is_frozen("policy"))

# 检查 active_models
training:
  stages:
    - active_models: ["policy"]  # 确保包含要训练的模型
```

### Q2: 找不到数据 / buffer 为空
```python
# 检查数据路径
data:
  demo_paths:
    - "/absolute/path/to/demos/*.hdf5"  # 使用绝对路径

# 检查是否加载成功
print(f"Demo buffer size: {len(data_hub.demo_buffer)}")
```

### Q3: CUDA out of memory
```yaml
# 减小 batch size
algorithm:
  batch_size: 32  # 降低到 32 或 16

# 或使用 CPU
device: "cpu"
```

### Q4: 自定义算法找不到模型
```python
class MyAlgo(BaseAlgorithm):
    REQUIRED_MODELS = ["policy", "q1"]  # 声明依赖
    
    def __init__(self, model_group, config):
        super().__init__(model_group, config)
        self._validate_model_group()  # 验证模型
```

---

## 📈 性能优化速查

### 数据加载优化
```yaml
data:
  load_images: false  # 不需要图像时关闭
  
algorithm:
  batch_size: 256  # GPU 利用率低时增大
```

### 训练速度优化
```python
# 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = algorithm.train_step(batch)
```

### 内存优化
```python
# 定期清空 rollout buffer
if len(data_hub.rollout_buffer) > max_size:
    data_hub.rollout_buffer.clear()
```

---

## 🔗 相关资源

- **完整文档**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **更新日志**: [CHANGELOG.md](CHANGELOG.md)
- **AI 助手指南**: [.github/copilot-instructions.md](.github/copilot-instructions.md)
- **GitHub**: https://github.com/hhzhsg/RL-unified-framework
- **Issues**: https://github.com/hhzhsg/RL-unified-framework/issues

---

**💡 提示**: 建议结合 IDE 的自动补全功能，所有接口都有完整的类型提示！
