"""
外部模型策略

接入非框架内的预训练模型（ACT、Pi0 等）
所有类都实现 Policy Protocol：act(), get_weights(), load_weights(), reset()

注意：部分模型需要在导入前设置 sys.path
"""

# Pi0 不需要特殊路径
from .pi0_policy import (
    Pi0Policy,
    Pi0Trainer,
    WeightSyncMode,
    create_pi0_policy_and_trainer,
    # 兼容旧名称
    Pi0PolicyAdapter,
    Pi0TrainerAdapter,
    create_pi0_adapters,
)

# ACT 需要延迟导入（依赖外部 detr 模块）
def get_act_policy():
    """
    延迟导入 ACTPolicy
    
    使用前确保 sys.path 包含 act-plus-plus 路径：
        sys.path.insert(0, '/path/to/act-plus-plus')
        ACTPolicy = get_act_policy()
    """
    from .act_policy import ACTPolicy
    return ACTPolicy


__all__ = [
    # Pi0
    "Pi0Policy",
    "Pi0Trainer",
    "WeightSyncMode",
    "create_pi0_policy_and_trainer",
    # ACT
    "get_act_policy",
    # 兼容
    "Pi0PolicyAdapter",
    "Pi0TrainerAdapter",
    "create_pi0_adapters",
]
