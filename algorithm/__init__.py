"""
算法模块

包含:
- BC: Behavior Cloning
- SAC: Soft Actor-Critic
- TD3BC: TD3 + BC
"""
from .base import BaseAlgorithm
from .bc import BC
from .sac import SAC
from .td3_bc import TD3BC

# 算法注册表
ALGORITHM_REGISTRY = {
    "bc": BC,
    "sac": SAC,
    "td3bc": TD3BC,
    "td3_bc": TD3BC,
}


def create_algorithm(name: str, model_group, config=None) -> BaseAlgorithm:
    """
    创建算法
    
    Args:
        name: 算法名称
        model_group: 模型组
        config: 算法配置
        
    Returns:
        算法实例
    """
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm: {name}. Available: {list(ALGORITHM_REGISTRY.keys())}")
    return ALGORITHM_REGISTRY[name](model_group, config)


__all__ = [
    "BaseAlgorithm",
    "BC",
    "SAC",
    "TD3BC",
    "ALGORITHM_REGISTRY",
    "create_algorithm",
]
