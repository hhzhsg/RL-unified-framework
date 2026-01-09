"""
VLA-RL Rollout Buffer
内存环形缓冲，FIFO 管理
"""
from __future__ import annotations
from typing import List, Optional
from collections import deque
import random
import numpy as np

from .base_buffer import BaseBuffer
from data import Transition, Episode


class RolloutBuffer(BaseBuffer):
    """
    Rollout 环形缓冲区
    
    特点:
    - 固定容量，FIFO 自动淘汰
    - 纯内存操作，高效
    - 训练结束后丢弃
    """
    
    def __init__(self, max_size: int = 100000):
        """
        Args:
            max_size: 最大 transition 数量
        """
        super().__init__(max_size)
        
        # 使用 deque 实现 FIFO
        self._transitions: deque[Transition] = deque(maxlen=max_size)
        
        # Episode 边界追踪 (用于 episode 采样)
        self._episode_boundaries: deque[int] = deque()  # 每个 episode 结束位置
        self._current_episode_start: int = 0
    
    def add_transition(self, transition: Transition):
        """添加单步数据"""
        self._transitions.append(transition)
        
        # 如果是 episode 结束，记录边界
        if transition.done:
            self._episode_boundaries.append(len(self._transitions))
            self._current_episode_start = len(self._transitions)
    
    def add_episode(self, episode: Episode):
        """添加完整 episode"""
        for t in episode.transitions:
            self._transitions.append(t)
        
        self._episode_boundaries.append(len(self._transitions))
        self._current_episode_start = len(self._transitions)
    
    def sample_transitions(self, batch_size: int) -> List[Transition]:
        """随机采样 transitions"""
        if len(self._transitions) == 0:
            return []
        
        # 直接从 deque 随机采样
        indices = [random.randint(0, len(self._transitions) - 1) 
                   for _ in range(batch_size)]
        return [self._transitions[i] for i in indices]
    
    def sample_episodes(self, batch_size: int) -> List[Episode]:
        """采样 episodes (从边界信息重建)"""
        # 简化实现：随机采样连续片段
        # 完整 episode 采样需要更多边界管理
        raise NotImplementedError("Use sample_transitions for rollout buffer")
    
    def __len__(self) -> int:
        return len(self._transitions)
    
    @property
    def num_episodes(self) -> int:
        return len(self._episode_boundaries)
    
    def _get_save_data(self):
        """Rollout 通常不保存，但保留接口"""
        return list(self._transitions)
    
    def _load_from_data(self, data):
        """加载数据"""
        self._transitions = deque(data, maxlen=self.max_size)
    
    def clear(self):
        """清空"""
        self._transitions.clear()
        self._episode_boundaries.clear()
        self._current_episode_start = 0
    
    def get_statistics(self) -> dict:
        """获取统计信息"""
        if len(self._transitions) == 0:
            return {"num_transitions": 0, "num_episodes": 0}
        
        # 统计 reward
        rewards = [t.reward for t in self._transitions]
        
        return {
            "num_transitions": len(self._transitions),
            "num_episodes": self.num_episodes,
            "capacity": self.max_size,
            "utilization": len(self._transitions) / self.max_size,
            "mean_reward": np.mean(rewards),
            "total_reward": np.sum(rewards),
        }
