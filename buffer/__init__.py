"""
Buffer 模块

提供三种数据缓冲区:
- ReplayBuffer: 通用经验回放 (用于 rollout: policy 实时推理的轨迹)
- HDF5DemoBuffer: HDF5 演示数据 (用于 demo: 预训练的专家数据)
- InterventionBuffer: 人工干预数据 (用于 intervention: rollout 中人工介入的纠正数据，持久化)
"""
from .base import BaseBuffer
from .replay import ReplayBuffer, SimpleReplayBuffer, RolloutBuffer
from .intervention import InterventionBuffer

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
    "InterventionBuffer",
    # HDF5
    "HDF5DemoBuffer",
    "inspect_hdf5",
    "HAS_HDF5",
]
