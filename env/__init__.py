"""
VLA-RL 环境模块
"""
from .base_env import BaseEnv
from .dummy_env import DummyEnv

# 环境注册表
ENV_REGISTRY = {
    "dummy": DummyEnv,
}


def create_env(config) -> BaseEnv:
    """根据配置创建环境"""
    from config import Config, EnvConfig
    
    if isinstance(config, Config):
        env_config = config.env
    elif isinstance(config, EnvConfig):
        env_config = config
    else:
        raise ValueError(f"Unknown config type: {type(config)}")
    
    env_cls = ENV_REGISTRY.get(env_config.name)
    if env_cls is None:
        raise ValueError(f"Unknown env: {env_config.name}. Available: {list(ENV_REGISTRY.keys())}")
    
    return env_cls(env_config)


def register_env(name: str, env_cls):
    """注册新环境"""
    ENV_REGISTRY[name] = env_cls
    return env_cls


__all__ = [
    "BaseEnv",
    "DummyEnv",
    "create_env",
    "register_env",
    "ENV_REGISTRY",
]
