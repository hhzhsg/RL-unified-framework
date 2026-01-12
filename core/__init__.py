"""
VLA-RL 核心模块

提供:
- TrainingLoop: 训练循环
- InferenceLoop: 推理循环
- WeightSync: 权重同步
- Stage: 训练阶段
"""
from .stage import Stage
from .weight_sync import (
    BaseWeightSync,
    QueueSync,
    SharedMemorySync,
    create_weight_sync,
    WEIGHT_SYNC_REGISTRY,
)
from .training_loop import TrainingLoop
from .inference_loop import InferenceLoop, HistoryBuffer

__all__ = [
    "Stage",
    "BaseWeightSync",
    "QueueSync",
    "SharedMemorySync",
    "create_weight_sync",
    "WEIGHT_SYNC_REGISTRY",
    "TrainingLoop",
    "InferenceLoop",
    "HistoryBuffer",
]
