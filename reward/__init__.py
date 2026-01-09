"""
VLA-RL Reward 模块

支持:
- 环境原始奖励
- Reward Shaping (Potential-based)
- Intrinsic Reward (RND, ICM)
- 多目标组合奖励
- Reward 归一化
"""
from .base_reward import BaseReward, RewardWrapper
from .env_reward import EnvReward
from .shaped_reward import PotentialShapingReward
from .intrinsic_reward import RNDReward
from .composite_reward import CompositeReward
from .normalizer import RewardNormalizer, RunningMeanStd

# 注册表
REWARD_REGISTRY = {
    "env": EnvReward,
    "shaped": PotentialShapingReward,
    "rnd": RNDReward,
    "composite": CompositeReward,
}


def create_reward(name: str, **kwargs) -> BaseReward:
    """
    创建 Reward 实例
    
    Args:
        name: 奖励类型名称
        **kwargs: 传递给构造函数的参数
        
    Returns:
        BaseReward 实例
    """
    if name not in REWARD_REGISTRY:
        raise ValueError(f"Unknown reward type: {name}. Available: {list(REWARD_REGISTRY.keys())}")
    return REWARD_REGISTRY[name](**kwargs)


__all__ = [
    "BaseReward",
    "RewardWrapper",
    "EnvReward",
    "PotentialShapingReward",
    "RNDReward",
    "CompositeReward",
    "RewardNormalizer",
    "RunningMeanStd",
    "REWARD_REGISTRY",
    "create_reward",
]
