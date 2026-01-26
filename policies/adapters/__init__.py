"""
策略适配器模块

提供将各种模型接入 HIL 框架的适配器

分类：
- StandardPolicyAdapter / AlgorithmAdapter: 框架内 Policy/Algorithm 的适配器
- Pi0PolicyAdapter / Pi0TrainerAdapter: VLA 大模型适配器（支持 LoRA 同步）
- SimpleMLPAdapter / SimpleMLPTrainer: 测试用简单适配器
"""
from .standard_adapter import StandardPolicyAdapter, AlgorithmAdapter
from .pi0_adapter import (
    Pi0PolicyAdapter,
    Pi0TrainerAdapter,
    create_pi0_adapters,
)
from .simple_mlp_adapter import SimpleMLPAdapter, SimpleMLPTrainer

__all__ = [
    # 标准适配器
    "StandardPolicyAdapter",
    "AlgorithmAdapter",
    # VLA 适配器
    "Pi0PolicyAdapter",
    "Pi0TrainerAdapter",
    "create_pi0_adapters",
    # 测试适配器
    "SimpleMLPAdapter",
    "SimpleMLPTrainer",
]
