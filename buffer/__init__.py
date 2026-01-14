"""
Buffer 模块

提供数据存储和采样:
- ReplayBuffer: 通用经验回放
- HDF5DemoBuffer: HDF5 演示数据
"""
from .base import BaseBuffer
from .replay import ReplayBuffer, SimpleReplayBuffer, RolloutBuffer

# HDF5 支持 (可选依赖)
try:
    from .hdf5 import HDF5DemoBuffer, inspect_hdf5
    HAS_HDF5 = True
except ImportError:
    HDF5DemoBuffer = None
    inspect_hdf5 = None
    HAS_HDF5 = False

__all__ = [
    "BaseBuffer",
    "ReplayBuffer",
    "SimpleReplayBuffer",
    "RolloutBuffer",
    # HDF5
    "HDF5DemoBuffer",
    "inspect_hdf5",
    "HAS_HDF5",
]
