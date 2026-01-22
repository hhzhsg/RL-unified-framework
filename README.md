# RL Framework

一个模块化、可扩展的强化学习框架。

## 设计原则

1. **模块自由组合**: Env/Buffer/Policy/Algorithm 可以任意搭配
2. **推理/训练分离**: 独立进程，互不干涉
3. **支持多场景**: Offline / Online / HIL
4. **最小修改成本**: 算法即插即用，不修改已有代码

## 目录结构

```
rl_framework/
├── core/                    # 系统主权层（接口定义，禁止写具体算法）
│   ├── interfaces/          # 抽象协议
│   ├── runtime/             # 运行时循环
│   ├── orchestration/       # 系统组装
│   └── synchronization/     # 同步机制
│
├── environments/            # 环境实现
├── data/                    # 数据层（Buffer, Sampler, Transform）
├── policies/                # 策略定义（Actor, Critic, Encoder）
├── algorithms/              # 训练算法（SAC, BC, HIL-SERL）
├── projects/                # 实验项目（隔离不同研究方向）
├── configs/                 # 可复用配置
├── scripts/                 # 命令行入口
├── utils/                   # 工具函数
└── common/                  # 共享类型
```

## 快速开始

```bash
# 使用配置文件训练
python scripts/train.py --config projects/hil_serl_example/config.yaml

# 或运行特定项目
cd projects/hil_serl_example
python run_experiment.py
```

## 添加新算法

1. 在 `algorithms/` 下创建新算法文件
2. 继承 `BaseAlgorithm` 或 `BaseOffPolicyAlgorithm`
3. 使用 `@register_algorithm("name")` 装饰器注册
4. 在 `policies/composed/` 下创建对应的策略组合（如需要）

## 添加新实验

1. 复制 `projects/_template/` 到新目录
2. 修改 `config.yaml`
3. 如需自定义组件，在项目目录下添加

## 核心接口

- `EnvInterface`: 环境接口
- `BufferInterface`: 缓冲区接口
- `SamplerInterface`: 采样器接口
- `PolicyInterface`: 策略接口
- `AlgorithmInterface`: 算法接口
