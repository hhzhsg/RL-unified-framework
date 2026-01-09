"""
Reward Shaping

基于势函数的奖励塑形，保证最优策略不变
参考: Ng et al., "Policy Invariance Under Reward Transformations" (1999)
"""
from typing import Optional, Dict, Any, Callable
import numpy as np

from .base_reward import BaseReward


class PotentialShapingReward(BaseReward):
    """
    Potential-based Reward Shaping
    
    r'(s, a, s') = r(s, a, s') + γ * φ(s') - φ(s)
    
    其中 φ 是势函数。这种形式保证最优策略不变。
    
    常见势函数：
    - 距离目标的负距离
    - 值函数估计
    - 手工设计的启发式
    """
    
    def __init__(self,
                 potential_fn: Callable[[np.ndarray], float],
                 gamma: float = 0.99,
                 scale: float = 1.0):
        """
        Args:
            potential_fn: 势函数 φ(s) -> float
            gamma: 折扣因子
            scale: 形状奖励缩放系数
        """
        super().__init__(name="shaped")
        self.potential_fn = potential_fn
        self.gamma = gamma
        self.scale = scale
    
    def compute(self,
                state: np.ndarray,
                action: np.ndarray,
                next_state: np.ndarray,
                env_reward: float,
                done: bool,
                info: Optional[Dict[str, Any]] = None) -> float:
        phi_s = self.potential_fn(state)
        phi_s_next = self.potential_fn(next_state) if not done else 0.0
        
        shaping = self.gamma * phi_s_next - phi_s
        return env_reward + self.scale * shaping


class DistanceShapingReward(BaseReward):
    """
    基于距离的奖励塑形
    
    势函数：φ(s) = -||s - goal||
    
    适用于目标导向任务
    """
    
    def __init__(self,
                 goal_state: np.ndarray,
                 gamma: float = 0.99,
                 scale: float = 1.0,
                 norm_type: str = "l2"):
        """
        Args:
            goal_state: 目标状态
            gamma: 折扣因子
            scale: 形状奖励缩放系数
            norm_type: 范数类型 "l1" | "l2"
        """
        super().__init__(name="distance_shaped")
        self.goal_state = np.asarray(goal_state)
        self.gamma = gamma
        self.scale = scale
        self.norm_type = norm_type
    
    def _distance(self, state: np.ndarray) -> float:
        """计算到目标的距离"""
        diff = state - self.goal_state
        if self.norm_type == "l1":
            return np.sum(np.abs(diff))
        else:  # l2
            return np.sqrt(np.sum(diff ** 2))
    
    def compute(self,
                state: np.ndarray,
                action: np.ndarray,
                next_state: np.ndarray,
                env_reward: float,
                done: bool,
                info: Optional[Dict[str, Any]] = None) -> float:
        # 势函数：负距离（越近势能越高）
        phi_s = -self._distance(state)
        phi_s_next = -self._distance(next_state) if not done else 0.0
        
        shaping = self.gamma * phi_s_next - phi_s
        return env_reward + self.scale * shaping


class ProgressReward(BaseReward):
    """
    进度奖励
    
    奖励 = 当前步进度 - 上一步进度
    
    适用于可以度量进度的任务（如接近目标、完成子目标等）
    """
    
    def __init__(self,
                 progress_fn: Callable[[np.ndarray], float],
                 scale: float = 1.0):
        """
        Args:
            progress_fn: 进度函数 (state) -> progress [0, 1]
            scale: 奖励缩放系数
        """
        super().__init__(name="progress")
        self.progress_fn = progress_fn
        self.scale = scale
    
    def compute(self,
                state: np.ndarray,
                action: np.ndarray,
                next_state: np.ndarray,
                env_reward: float,
                done: bool,
                info: Optional[Dict[str, Any]] = None) -> float:
        progress_prev = self.progress_fn(state)
        progress_curr = self.progress_fn(next_state)
        
        progress_reward = (progress_curr - progress_prev) * self.scale
        return env_reward + progress_reward
