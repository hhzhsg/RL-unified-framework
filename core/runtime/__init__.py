"""
Runtime 模块

运行时循环
"""
from .base_loop import BaseLoop
from .training_loop import TrainingLoop
from .inference_loop import InferenceLoop
from .evaluation_loop import EvaluationLoop

# HIL 组件（模型无关）
from .hil_actor_loop import HILActorLoop, HILActorConfig
from .hil_learner_loop import HILLearnerLoop, HILLearnerConfig
from .hil_trainer import HILTrainer

# 向后兼容别名
HILSerlActorLoop = HILActorLoop
HILSerlActorConfig = HILActorConfig
HILSerlLearnerLoop = HILLearnerLoop
HILSerlLearnerConfig = HILLearnerConfig
HILSerlTrainer = HILTrainer

__all__ = [
    # 基础循环
    "BaseLoop",
    "TrainingLoop",
    "InferenceLoop",
    "EvaluationLoop",
    # HIL 组件
    "HILActorLoop",
    "HILActorConfig",
    "HILLearnerLoop",
    "HILLearnerConfig",
    "HILTrainer",
    # 向后兼容
    "HILSerlActorLoop",
    "HILSerlActorConfig",
    "HILSerlLearnerLoop",
    "HILSerlLearnerConfig",
    "HILSerlTrainer",
]
