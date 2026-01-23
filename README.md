# RL Unified Framework

一个模块化、可扩展的强化学习框架，专注于真机 RL 和 Human-in-the-Loop 训练。

## 更新历史

- 2026-01-23: HIL-SERL 框架迁移，验证 Franka pick-and-place 60% 成功率
- 2026-01-22: 初始版本，SAC/BC 算法实现

## 设计目标

1. **模块自由组合**: Env/Buffer/Policy/Algorithm 可以任意搭配
2. **推理/训练分离**: 独立进程，互不干涉
3. **支持多场景**: Offline / Online / HIL（人机交互）
4. **最小修改成本**: 算法即插即用，不修改已有代码

## 环境安装

```bash
# 克隆仓库
git clone <repo_url>
cd RL-unified-framework

# 安装依赖
pip install -r requirements.txt

# （可选）安装开发依赖
pip install -e .
```

## 目录结构

```
RL-unified-framework/
├── core/                    # 框架层（接口定义，禁止写具体算法）
│   ├── interfaces/          # 抽象协议
│   ├── runtime/             # 运行时循环（TrainingLoop, HILActorLoop等）
│   ├── orchestration/       # 组件注册与系统组装
│   └── synchronization/     # Actor-Learner 同步机制
│
├── algorithms/              # 训练算法（SAC, BC）
├── policies/                # 策略网络
│   ├── components/          # 原子组件（Actor, Critic, Encoder）
│   ├── composed/            # 组合策略（SACPolicy, HILSERLPolicy）
│   └── adapters/            # 模型适配器（Pi0Adapter等）
├── data/                    # 数据层
│   ├── buffers/             # ReplayBuffer, InterventionBuffer
│   ├── samplers/            # UniformSampler, HILSERLSampler
│   └── transforms/          # 数据预处理
├── env/                     # 环境封装
├── projects/                # 实验项目（隔离不同研究方向）
├── configs/                 # 可复用配置
├── scripts/                 # 命令行入口
├── utils/                   # 工具函数
└── common/                  # 共享类型定义
```

## 使用方式

### 快速开始

```bash
# 使用配置文件训练
python scripts/train.py --config projects/_template/config.yaml --steps 100000

# 推理
python scripts/infer.py --checkpoint checkpoints/step_50000.pt
```

### HIL 训练（Human-in-the-Loop）

```python
from core.runtime import HILTrainer
from core.interfaces.adapters import StandardPolicyAdapter

# 本地调试模式
trainer = HILTrainer(
    actor_adapter=StandardPolicyAdapter(policy),
    learner_adapter=AlgorithmAdapter(algorithm),
    env=env,
    config=config,
    mode="local",
)
results = trainer.run_local(num_steps=10000)
```

### 添加新算法

```python
from core.orchestration import register_algorithm
from algorithms.base_algorithm import BaseOffPolicyAlgorithm

@register_algorithm("my_algo")
class MyAlgorithm(BaseOffPolicyAlgorithm):
    def update(self, batch):
        # 实现训练逻辑
        return {"loss": loss.item()}
```

## 核心接口

| 接口 | 说明 |
|------|------|
| `EnvInterface` | 环境接口（reset, step） |
| `BufferInterface` | 缓冲区接口（add, sample） |
| `SamplerInterface` | 采样器接口（混合多源数据） |
| `PolicyInterface` | 策略接口（act, forward） |
| `AlgorithmInterface` | 算法接口（update, save, load） |
| `PolicyAdapter` | 策略适配器（解耦 HIL 与具体模型） |

## 架构特点

### HIL 与 Model 解耦

```
HILActorLoop / HILLearnerLoop
    └── PolicyAdapterProtocol  ← 抽象协议
          ├── StandardPolicyAdapter (SAC/BC)
          ├── Pi0PolicyAdapter (pi0.5/OpenVLA, LoRA同步)
          └── 可扩展任意模型
```

支持只同步 LoRA/Adapter 参数，适配大型 VLA 模型。

### 三源数据管理

- **demo**: 离线专家数据（HDF5）
- **rollout**: 在线 policy 轨迹
- **intervention**: 人工纠正数据（2x 采样权重）
