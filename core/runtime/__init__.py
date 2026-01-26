"""
Runtime 模块 - 运行时循环

业界标准：Actor-Learner-Evaluator 架构

核心组件：
- ActorLoop: 环境交互，收集数据（Online RL）
- LearnerLoop: 从数据学习，更新策略（Offline/Online RL）
- EvaluatorLoop: 评估策略性能

HIL 扩展（分布式训练）：
- HILActorLoop: 带人类干预的 Actor（VR/SpaceMouse）
- HILLearnerLoop: 带数据分流的 Learner（intervention 2x 权重）

场景选择：
- Offline RL: LearnerLoop
- Online RL: ActorLoop + LearnerLoop
- HIL: HILActorLoop + HILLearnerLoop（两个独立进程）
- 评估: EvaluatorLoop

使用示例：
    # 普通训练
    python scripts/train.py --config xxx.yaml
    
    # HIL 分布式训练（两个终端）
    python scripts/train.py --config xxx.yaml --role learner
    python scripts/train.py --config xxx.yaml --role actor
"""
from .base_loop import BaseLoop

# 标准 Loop
from .actor_loop import ActorLoop
from .learner_loop import LearnerLoop
from .evaluator_loop import EvaluatorLoop

# HIL Loop（分布式）
from .hil_loop import (
    HILActorLoop,
    HILActorConfig,
    HILLearnerLoop,
    HILLearnerConfig,
)

__all__ = [
    # 基类
    "BaseLoop",
    
    # 标准 Loop
    "ActorLoop",
    "LearnerLoop",
    "EvaluatorLoop",
    
    # HIL Loop
    "HILActorLoop",
    "HILActorConfig",
    "HILLearnerLoop",
    "HILLearnerConfig",
]

