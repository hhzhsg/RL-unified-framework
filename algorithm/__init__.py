"""
VLA-RL 算法模块

所有算法继承 BaseAlgorithm，实现 train_step() 方法
通过 ALGORITHM_REGISTRY 注册，支持配置驱动创建
"""
from .base_algorithm import BaseAlgorithm
from .bc import BC
from .sac import SAC
from .td3_bc import TD3BC
from .recap import RECAPAlgorithm, RECAPTrainer, create_recap_models

# 算法注册表
ALGORITHM_REGISTRY = {
    "bc": BC,
    "sac": SAC,
    "td3_bc": TD3BC,
    "recap": RECAPAlgorithm,
}


def create_algorithm(name: str, model_group, config=None, **kwargs) -> BaseAlgorithm:
    """
    创建算法实例
    
    Args:
        name: 算法名称 (bc, sac, td3_bc, ...)
        model_group: ModelGroup 实例
        config: AlgorithmConfig 配置
        **kwargs: 额外参数透传给算法构造函数
        
    Returns:
        BaseAlgorithm 实例
    """
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown algorithm: {name}. "
            f"Available: {list(ALGORITHM_REGISTRY.keys())}"
        )
    
    return ALGORITHM_REGISTRY[name](model_group, config, **kwargs)


def register_algorithm(name: str, algo_cls):
    """
    注册新算法
    
    Example:
        @register_algorithm("my_algo")
        class MyAlgo(BaseAlgorithm):
            ...
    """
    ALGORITHM_REGISTRY[name] = algo_cls
    return algo_cls


__all__ = [
    # 基类
    "BaseAlgorithm",
    # 算法实现
    "BC",
    "SAC",
    "TD3BC",
    "RECAPAlgorithm",
    "RECAPTrainer",
    # 工厂函数
    "create_algorithm",
    "register_algorithm",
    "create_recap_models",
    "ALGORITHM_REGISTRY",
]
