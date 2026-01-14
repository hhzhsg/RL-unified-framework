"""
策略模块

包含:
- BasePolicy: 策略基类
- MLPPolicy: 确定性 MLP 策略
- MLPGaussianPolicy: 随机 MLP 策略 (SAC 用)
- ResidualPolicy: 残差策略
- EnsemblePolicy: 集成策略
"""
from .base import BasePolicy
from .mlp import MLPPolicy, MLPGaussianPolicy
from .composite import ResidualPolicy, EnsemblePolicy

# 策略注册表
POLICY_REGISTRY = {
    "mlp": MLPPolicy,
    "mlp_gaussian": MLPGaussianPolicy,
}

__all__ = [
    "BasePolicy",
    "MLPPolicy",
    "MLPGaussianPolicy",
    "ResidualPolicy",
    "EnsemblePolicy",
    "POLICY_REGISTRY",
]
