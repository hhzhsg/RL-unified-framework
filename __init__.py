"""
VLA-RL: Vision-Language-Action Reinforcement Learning Framework

统一的机器人强化学习框架，支持:
- 离线学习 (BC, TD3+BC)
- 在线学习 (SAC)
- 推理/训练分离
- 多种机器人平台

模块结构:
- policy/: 策略网络
- network/: 基础网络 (MLP, Q网络)
- algorithm/: 训练算法
- buffer/: 数据缓冲区
- env/: 环境接口
- data/: 数据类型和处理
- core/: 核心组件 (训练循环, 权重同步)
- robot/: 机器人适配器
- config/: 配置管理
- utils/: 工具函数

Quick Start:
    from policy import MLPPolicy
    from algorithm import BC, SAC
    from core import ModelGroup, TrainingLoop
"""

__version__ = "0.1.0"
