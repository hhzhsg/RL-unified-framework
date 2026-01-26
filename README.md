# RL Unified Framework

一个模块化、可扩展的强化学习框架，专注于真机 RL 和 Human-in-the-Loop 训练。

## 设计目标

1. **模块自由组合**: Env/Buffer/Policy/Algorithm 可以任意搭配
2. **推理/训练分离**: 独立进程，互不干涉
3. **支持多场景**: Offline / Online / HIL（人机交互）
4. **最小修改成本**: 算法即插即用，不修改已有代码

## 环境安装

```bash
# 安装依赖
pip install -r requirements.txt
```

## 目录结构

```
RL-unified-framework/
├── core/                    # 框架层（接口定义，禁止写具体算法）
│   ├── interfaces/          # 抽象协议
│   ├── runtime/             # 运行时循环（Actor/Learner/Evaluator）
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
│   ├── dummy_env/           # 测试用虚拟环境
│   ├── h1_robot/            # H1 机器人环境
│   └── wrappers/            # 干预设备 Wrapper（VRWrapper等）
├── projects/                # 实验项目（隔离不同研究方向）
├── scripts/                 # 命令行入口
└── utils/                   # 工具函数
```

## 模块职责速查

| 目录 | 回答问题 | 示例 |
|------|----------|------|
| `core/runtime/` | **怎么跑**（Actor/Learner 循环） | `HILActorLoop`, `HILLearnerLoop` |
| `core/synchronization/` | **怎么通信**（gRPC/Queue） | `LearnerServer`, `ActorClient` |
| `env/wrappers/` | **怎么干预**（VR 接管检测） | `VRWrapper` |
| `data/buffers/` | **数据存哪**（三类数据源） | `DemoBuffer`, `InterventionBuffer` |
| `data/samplers/` | **怎么采样**（加权采样） | `HILSERLSampler`（intervention 2x） |
| `algorithms/` | **怎么更新**（loss + 优化器） | `SACAlgorithm.update()` |
| `policies/composed/` | **网络长什么样**（组合结构） | `SACPolicy` = Actor + Critic |
| `policies/components/` | **网络怎么实现**（原子模块） | `GaussianActor`, `QCritic` |
| `policies/adapters/` | **怎么统一接口**（适配不同模型） | `StandardPolicyAdapter`, `Pi0Adapter` |

## 使用方式

### 快速开始

```bash
# 使用配置文件训练
python scripts/train.py --config projects/_template/config.yaml --steps 10000

# 评估
python scripts/eval.py --config projects/_template/config.yaml --checkpoint checkpoints/step_10000.pt
```

### HIL 训练（Human-in-the-Loop）

HIL 使用分布式 Actor-Learner 架构：

```bash
# 终端 1：启动 Learner（GPU 服务器）
python scripts/train.py --config projects/h1_hil/config.yaml --role learner

# 终端 2：启动 Actor（机器人端）
python scripts/train.py --config projects/h1_hil/config.yaml --role actor
```

```python
# 或者代码中直接使用
from core.runtime import HILActorLoop, HILLearnerLoop
from core.synchronization import create_actor_client, create_learner_server

# Actor 端
actor = HILActorLoop(
    policy_adapter=adapter,
    env=env,
    config=actor_config,
    sync_config=sync_config,
    mode="grpc",
)
actor.run(num_steps=10000)
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

## 运行时循环（Actor-Learner-Evaluator 架构）

采用业界标准的 Actor-Learner 分离架构：

| Loop | 职责 | 适用场景 |
|------|------|---------|
| `ActorLoop` | 环境交互，收集数据 | Online RL |
| `LearnerLoop` | 从数据学习，更新策略 | Offline/Online RL |
| `EvaluatorLoop` | 评估策略性能 | 策略评估 |
| `HILActorLoop` | 带人类干预的 Actor | HIL 训练 |
| `HILLearnerLoop` | 带数据分流的 Learner | HIL 训练 |

场景选择：
- **Offline RL**: `LearnerLoop`
- **Online RL**: `ActorLoop` + `LearnerLoop`
- **HIL**: `HILActorLoop` + `HILLearnerLoop`
- **评估**: `EvaluatorLoop`

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
