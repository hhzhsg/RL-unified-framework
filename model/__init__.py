from .base_policy import BasePolicy
from .model_group import ModelGroup
from .mlp_policy import MLPPolicy, MLPGaussianPolicy
from .composite_policy import ResidualPolicy, EnsemblePolicy

# 策略注册表
POLICY_REGISTRY = {
    "mlp": MLPPolicy,
    "mlp_gaussian": MLPGaussianPolicy,
}


def create_policy(policy_type: str, **kwargs) -> BasePolicy:
    """创建策略"""
    if policy_type not in POLICY_REGISTRY:
        raise ValueError(f"Unknown policy: {policy_type}. Available: {list(POLICY_REGISTRY.keys())}")
    
    return POLICY_REGISTRY[policy_type](**kwargs)


def register_policy(name: str, policy_cls):
    """注册新策略"""
    POLICY_REGISTRY[name] = policy_cls


__all__ = [
    "BasePolicy",
    "ModelGroup",
    "MLPPolicy",
    "MLPGaussianPolicy",
    "ResidualPolicy",
    "EnsemblePolicy",
    "create_policy",
    "register_policy",
    "POLICY_REGISTRY",
]
