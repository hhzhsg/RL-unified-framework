"""
环境模块

包含:
- BaseEnv: 环境基类
- DummyEnv: 测试环境
"""
from .base import BaseEnv
from .dummy import DummyEnv

# 环境注册表
ENV_REGISTRY = {
    "dummy": DummyEnv,
}


def create_env(config) -> BaseEnv:
    """
    创建环境
    
    Args:
        config: 环境配置
        
    Returns:
        环境实例
    """
    env_name = config.name if hasattr(config, 'name') else "dummy"
    if env_name not in ENV_REGISTRY:
        raise ValueError(f"Unknown env: {env_name}. Available: {list(ENV_REGISTRY.keys())}")
    return ENV_REGISTRY[env_name](config)


__all__ = [
    "BaseEnv",
    "DummyEnv",
    "ENV_REGISTRY",
    "create_env",
]
