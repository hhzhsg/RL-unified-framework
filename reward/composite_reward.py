"""
组合奖励

支持多个奖励函数的加权组合
"""
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from .base_reward import BaseReward
from data import Batch


class CompositeReward(BaseReward):
    """
    组合多个奖励函数
    
    r_total = Σ w_i * r_i(s, a, s')
    
    使用场景：
    - 多目标强化学习
    - 结合外在和内在奖励
    - 奖励消融实验
    """
    
    def __init__(self, 
                 rewards: Optional[List[Tuple[BaseReward, float]]] = None,
                 normalize_weights: bool = False):
        """
        Args:
            rewards: [(reward_fn, weight), ...] 奖励函数和权重列表
            normalize_weights: 是否归一化权重使其和为 1
        """
        super().__init__(name="composite")
        self.rewards: List[Tuple[BaseReward, float]] = rewards or []
        self.normalize_weights = normalize_weights
        
        if normalize_weights and self.rewards:
            self._normalize()
    
    def _normalize(self):
        """归一化权重"""
        total = sum(w for _, w in self.rewards)
        if total > 0:
            self.rewards = [(r, w / total) for r, w in self.rewards]
    
    def add(self, reward: BaseReward, weight: float = 1.0):
        """添加奖励函数"""
        self.rewards.append((reward, weight))
        if self.normalize_weights:
            self._normalize()
    
    def compute(self,
                state: np.ndarray,
                action: np.ndarray,
                next_state: np.ndarray,
                env_reward: float,
                done: bool,
                info: Optional[Dict[str, Any]] = None) -> float:
        total_reward = 0.0
        
        for reward_fn, weight in self.rewards:
            r = reward_fn.compute(state, action, next_state, env_reward, done, info)
            total_reward += weight * r
        
        return total_reward
    
    def transform_batch(self, batch: Batch) -> Batch:
        """批量转换（使用第一个奖励函数的结果作为基础）"""
        if not self.rewards:
            return batch
        
        import copy
        new_batch = copy.copy(batch)
        
        # 累加所有奖励
        total_rewards = np.zeros(len(batch), dtype=np.float32)
        for reward_fn, weight in self.rewards:
            transformed = reward_fn.transform_batch(batch)
            total_rewards += weight * np.asarray(transformed.reward)
        
        new_batch.reward = total_rewards
        return new_batch
    
    def reset(self):
        for reward_fn, _ in self.rewards:
            reward_fn.reset()
    
    def update(self, batch: Batch):
        for reward_fn, _ in self.rewards:
            reward_fn.update(batch)
    
    def get_stats(self) -> Dict[str, float]:
        stats = {"call_count": self._call_count}
        for i, (reward_fn, weight) in enumerate(self.rewards):
            sub_stats = reward_fn.get_stats()
            for k, v in sub_stats.items():
                stats[f"{reward_fn.name}_{i}/{k}"] = v
        return stats


class ScheduledReward(BaseReward):
    """
    带调度的奖励
    
    支持奖励权重随训练进度变化
    """
    
    def __init__(self,
                 reward: BaseReward,
                 schedule_fn,  # (step) -> weight
                 initial_step: int = 0):
        """
        Args:
            reward: 基础奖励函数
            schedule_fn: 调度函数 (step) -> weight
            initial_step: 初始步数
        """
        super().__init__(name="scheduled")
        self.reward = reward
        self.schedule_fn = schedule_fn
        self.step = initial_step
    
    def compute(self,
                state: np.ndarray,
                action: np.ndarray,
                next_state: np.ndarray,
                env_reward: float,
                done: bool,
                info: Optional[Dict[str, Any]] = None) -> float:
        weight = self.schedule_fn(self.step)
        r = self.reward.compute(state, action, next_state, env_reward, done, info)
        self.step += 1
        return weight * r
    
    def reset(self):
        self.reward.reset()
    
    def set_step(self, step: int):
        """设置当前步数"""
        self.step = step
