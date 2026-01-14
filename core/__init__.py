"""
核心模块

包含:
- ModelGroup: 模型组管理
- TrainingLoop: 训练循环
- InferenceLoop: 推理循环
- WeightSync: 权重同步
"""
from .model_group import ModelGroup
from .training_loop import TrainingLoop
from .inference_loop import InferenceLoop
from .weight_sync import (
    BaseWeightSync,
    SharedMemorySync,
    QueueSync,
    create_weight_sync,
    SYNC_REGISTRY,
)

__all__ = [
    "ModelGroup",
    "TrainingLoop",
    "InferenceLoop",
    "BaseWeightSync",
    "SharedMemorySync",
    "QueueSync",
    "create_weight_sync",
    "SYNC_REGISTRY",
]
