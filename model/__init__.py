"""
VLA-RL 模型模块

包含:
- BasePolicy: 策略基类
- MLPPolicy: 确定性 MLP 策略
- MLPGaussianPolicy: 随机 MLP 策略 (SAC 用)
- QNetwork: Q 网络
- VNetwork: V 网络
- ModelGroup: 模型组管理
- VLA 模块: π0, π0.5, π0.6* (RECAP)
"""
from .base_policy import BasePolicy
from .mlp_policy import MLPPolicy, MLPGaussianPolicy
from .q_network import QNetwork, VNetwork
from .model_group import ModelGroup

# VLA 模块
from . import vla
from .vla import (
    PI0Config,
    PI05Config,
    ValueConfig,
    RECAPConfig,
    PI0Policy,
    PI05Policy,
    ValueFunction,
)

# 策略注册表
POLICY_REGISTRY = {
    "mlp": MLPPolicy,
    "mlp_gaussian": MLPGaussianPolicy,
    "pi0": PI0Policy,
    "pi05": PI05Policy,
}

__all__ = [
    # 基础策略
    "BasePolicy",
    "MLPPolicy",
    "MLPGaussianPolicy",
    "QNetwork",
    "VNetwork",
    "ModelGroup",
    "POLICY_REGISTRY",
    # VLA 模块
    "vla",
    "PI0Config",
    "PI05Config",
    "ValueConfig",
    "RECAPConfig",
    "PI0Policy",
    "PI05Policy",
    "ValueFunction",
]
