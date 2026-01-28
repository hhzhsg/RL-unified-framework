"""
策略适配器模块

提供将各种模型接入 HIL 框架的适配器

分类：
- StandardPolicyAdapter / AlgorithmAdapter: 框架内 Policy/Algorithm 的适配器
- Pi0PolicyAdapter / Pi0TrainerAdapter: VLA 大模型适配器（支持 LoRA 同步）
- SimpleMLPAdapter / SimpleMLPTrainer: 测试用简单适配器
- ACTPolicy: ACT++ 模型适配器（需要预先设置 sys.path）
"""
from .standard_adapter import StandardPolicyAdapter, AlgorithmAdapter
from .pi0_adapter import (
    Pi0PolicyAdapter,
    Pi0TrainerAdapter,
    create_pi0_adapters,
)
from .simple_mlp_adapter import SimpleMLPAdapter, SimpleMLPTrainer

# ACTPolicy 需要延迟导入，因为依赖外部 detr 模块
# 使用时确保 sys.path 已包含 act-plus-plus 路径
def get_act_policy():
    """延迟导入 ACTPolicy（需要先设置 sys.path）"""
    from .act_adapter import ACTPolicy
    return ACTPolicy

__all__ = [
    # 标准适配器
    "StandardPolicyAdapter",
    "AlgorithmAdapter",
    # VLA 适配器
    "Pi0PolicyAdapter",
    "Pi0TrainerAdapter",
    "create_pi0_adapters",
    # ACT 适配器（延迟导入）
    "get_act_policy",
    # 测试适配器
    "SimpleMLPAdapter",
    "SimpleMLPTrainer",
]
