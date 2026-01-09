"""
VLA-RL Rollout Buffer
内存环形缓冲，FIFO 管理
"""
from typing import List
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
    - 用于 Online RL 数据收集
    """
    
    def __init__(self, max_size: int = 100000):
        super().__init__(max_size)
        
        # 使用 deque 实现 FIFO
        self._transitions: deque = deque(maxlen=max_size)
        
        # Episode 边界追踪
        self._episode_boundaries: deque = deque()
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
        
        indices = [random.randint(0, len(self._transitions) - 1) 
                   for _ in range(batch_size)]
        return [self._transitions[i] for i in indices]
    
    def sample_episodes(self, batch_size: int) -> List[Episode]:
        """采样 episodes"""
        raise NotImplementedError("Use sample_transitions for rollout buffer")
    
    def __len__(self) -> int:
        return len(self._transitions)
    
    @property
    def num_episodes(self) -> int:
        return len(self._episode_boundaries)
    
    def _get_save_data(self):
        return list(self._transitions)
    
    def _load_from_data(self, data):
        self._transitions = deque(data, maxlen=self.max_size)
    
    def clear(self):
        self._transitions.clear()
        self._episode_boundaries.clear()
        self._current_episode_start = 0
    
    def get_statistics(self) -> dict:
        if len(self._transitions) == 0:
            return {"num_transitions": 0, "num_episodes": 0}
        
        rewards = [t.reward for t in self._transitions]
        
        return {
            "num_transitions": len(self._transitions),
            "num_episodes": self.num_episodes,
            "capacity": self.max_size,
            "utilization": len(self._transitions) / self.max_size,
            "mean_reward": np.mean(rewards),
            "total_reward": np.sum(rewards),
        }
