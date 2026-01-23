"""Environments模块"""
from .base_env import BaseEnv
from .dummy_env import DummyEnv

__all__ = ["BaseEnv", "DummyEnv"]

# H1 机器人环境（可选，需要硬件 SDK）
try:
    from .h1_robot import H1RobotEnv
    __all__.append("H1RobotEnv")
except ImportError:
    H1RobotEnv = None
