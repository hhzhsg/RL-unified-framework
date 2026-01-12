"""
VLA-RL 模型模块

包含:
- BasePolicy: 策略基类
- MLPPolicy: 确定性 MLP 策略
- MLPGaussianPolicy: 随机 MLP 策略 (SAC 用)
- QNetwork: Q 网络
- VNetwork: V 网络
- ModelGroup: 模型组管理
"""
from .base_policy import BasePolicy
from .mlp_policy import MLPPolicy, MLPGaussianPolicy
from .q_network import QNetwork, VNetwork
from .model_group import ModelGroup

# 策略注册表
POLICY_REGISTRY = {
    "mlp": MLPPolicy,
    "mlp_gaussian": MLPGaussianPolicy,
}

__all__ = [
    "BasePolicy",
    "MLPPolicy",
    "MLPGaussianPolicy",
    "QNetwork",
    "VNetwork",
    "ModelGroup",
    "POLICY_REGISTRY",
]
