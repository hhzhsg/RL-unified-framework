from .base_algorithm import BaseAlgorithm
from .bc import BC
from .sac import SAC, QNetwork
from .td3_bc import TD3BC

# 算法注册表
ALGORITHM_REGISTRY = {
    "bc": BC,
    "sac": SAC,
    "td3_bc": TD3BC,
}


def create_algorithm(name: str, model_group, config=None, **kwargs) -> BaseAlgorithm:
    """创建算法"""
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm: {name}. Available: {list(ALGORITHM_REGISTRY.keys())}")
    
    return ALGORITHM_REGISTRY[name](model_group, config, **kwargs)


def register_algorithm(name: str, algo_cls):
    """注册新算法"""
    ALGORITHM_REGISTRY[name] = algo_cls


__all__ = [
    "BaseAlgorithm",
    "BC",
    "SAC",
    "QNetwork",
    "create_algorithm",
    "register_algorithm",
    "ALGORITHM_REGISTRY",
]
