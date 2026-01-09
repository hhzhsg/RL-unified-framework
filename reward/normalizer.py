"""
Reward 归一化

Running statistics for reward normalization
"""
from typing import Optional
import numpy as np


class RunningMeanStd:
    """
    运行时均值和标准差统计
    
    使用 Welford's online algorithm 进行数值稳定的更新
    """
    
    def __init__(self, shape: tuple = (), epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        self.epsilon = epsilon
    
    def update(self, x: np.ndarray):
        """更新统计量"""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if len(x.shape) > 0 else 1
        
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        """从批次统计量更新"""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        
        self.mean = new_mean
        self.var = M2 / total_count
        self.count = total_count
    
    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + self.epsilon)
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """归一化"""
        return (x - self.mean) / self.std
    
    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """反归一化"""
        return x * self.std + self.mean


class RewardNormalizer:
    """
    奖励归一化器
    
    维护奖励的运行统计并进行归一化
    """
    
    def __init__(self, 
                 clip_range: float = 10.0,
                 gamma: float = 0.99,
                 per_episode: bool = True,
                 epsilon: float = 1e-8):
        """
        Args:
            clip_range: 归一化后的裁剪范围 [-clip_range, clip_range]
            gamma: 折扣因子（用于 return 归一化）
            per_episode: 是否按 episode 归一化 return
            epsilon: 数值稳定性常数
        """
        self.clip_range = clip_range
        self.gamma = gamma
        self.per_episode = per_episode
        self.epsilon = epsilon
        
        # 奖励统计
        self.reward_stats = RunningMeanStd(epsilon=epsilon)
        
        # Return 统计（用于 GAE 等）
        self.return_stats = RunningMeanStd(epsilon=epsilon)
        
        # Episode return 累积
        self._episode_returns: list = []
        self._current_return: float = 0.0
    
    def normalize_reward(self, reward: float, update: bool = True) -> float:
        """
        归一化单步奖励
        
        Args:
            reward: 原始奖励
            update: 是否更新统计量
        """
        if update:
            self.reward_stats.update(np.array([reward]))
        
        normalized = (reward - self.reward_stats.mean) / self.reward_stats.std
        return np.clip(normalized, -self.clip_range, self.clip_range)
    
    def normalize_rewards(self, rewards: np.ndarray, update: bool = True) -> np.ndarray:
        """
        批量归一化奖励
        """
        if update:
            self.reward_stats.update(rewards)
        
        normalized = self.reward_stats.normalize(rewards)
        return np.clip(normalized, -self.clip_range, self.clip_range)
    
    def update_episode(self, episode_rewards: np.ndarray):
        """
        更新 episode 统计
        
        Args:
            episode_rewards: 一个 episode 的所有奖励
        """
        # 计算 discounted return
        returns = []
        G = 0.0
        for r in reversed(episode_rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        # 更新 return 统计
        self.return_stats.update(np.array(returns))
    
    def normalize_returns(self, returns: np.ndarray) -> np.ndarray:
        """归一化 returns"""
        normalized = self.return_stats.normalize(returns)
        return np.clip(normalized, -self.clip_range, self.clip_range)
    
    def reset(self):
        """重置 episode 累积"""
        self._current_return = 0.0
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "reward_mean": float(self.reward_stats.mean),
            "reward_std": float(self.reward_stats.std),
            "reward_count": self.reward_stats.count,
            "return_mean": float(self.return_stats.mean),
            "return_std": float(self.return_stats.std),
        }
