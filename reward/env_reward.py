"""
环境原始奖励
"""
from typing import Optional, Dict, Any
import numpy as np

from .base_reward import BaseReward


class EnvReward(BaseReward):
    """
    直接使用环境原始奖励
    
    这是最简单的奖励类型，适用于：
    - BC (不使用奖励)
    - 环境奖励设计良好的情况
    """
    
    def __init__(self, scale: float = 1.0, shift: float = 0.0):
        """
        Args:
            scale: 奖励缩放系数
            shift: 奖励偏移
        """
        super().__init__(name="env")
        self.scale = scale
        self.shift = shift
    
    def compute(self,
                state: np.ndarray,
                action: np.ndarray,
                next_state: np.ndarray,
                env_reward: float,
                done: bool,
                info: Optional[Dict[str, Any]] = None) -> float:
        """直接返回环境奖励（可选缩放和偏移）"""
        return env_reward * self.scale + self.shift
    
    def transform_batch(self, batch):
        """向量化版本"""
        import copy
        new_batch = copy.copy(batch)
        new_batch.reward = batch.reward * self.scale + self.shift
        return new_batch


class SparseReward(BaseReward):
    """
    稀疏奖励
    
    只在特定条件下给予奖励，例如：
    - 任务成功时 +1
    - 其他时候 0
    """
    
    def __init__(self, 
                 success_reward: float = 1.0,
                 failure_reward: float = 0.0,
                 step_penalty: float = 0.0):
        """
        Args:
            success_reward: 成功时的奖励
            failure_reward: 失败时的奖励
            step_penalty: 每步惩罚（鼓励快速完成）
        """
        super().__init__(name="sparse")
        self.success_reward = success_reward
        self.failure_reward = failure_reward
        self.step_penalty = step_penalty
    
    def compute(self,
                state: np.ndarray,
                action: np.ndarray,
                next_state: np.ndarray,
                env_reward: float,
                done: bool,
                info: Optional[Dict[str, Any]] = None) -> float:
        reward = self.step_penalty
        
        if done and info is not None:
            if info.get("success", False):
                reward += self.success_reward
            else:
                reward += self.failure_reward
        
        return reward


class ClippedReward(BaseReward):
    """
    裁剪奖励
    
    将奖励限制在 [min_reward, max_reward] 范围内
    """
    
    def __init__(self, 
                 min_reward: float = -10.0,
                 max_reward: float = 10.0):
        super().__init__(name="clipped")
        self.min_reward = min_reward
        self.max_reward = max_reward
    
    def compute(self,
                state: np.ndarray,
                action: np.ndarray,
                next_state: np.ndarray,
                env_reward: float,
                done: bool,
                info: Optional[Dict[str, Any]] = None) -> float:
        return np.clip(env_reward, self.min_reward, self.max_reward)
