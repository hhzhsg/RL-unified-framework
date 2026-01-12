"""
VLA-RL Rollout Buffer

FIFO 环形缓冲区，用于存储在线采集的数据
"""
from typing import List, Optional
from collections import deque
import random

from .base_buffer import BaseBuffer
from data import Transition, Episode


class RolloutBuffer(BaseBuffer):
    """
    Rollout Buffer - FIFO 环形缓冲区
    
    用于存储策略在线采集的数据
    当容量满时，自动移除最旧的数据
    """
    
    def __init__(self, max_size: int = 100000):
        super().__init__(max_size)
        self._transitions: deque = deque(maxlen=max_size)
        self._episodes: List[Episode] = []
        self._current_episode: Optional[Episode] = None
    
    def add_transition(self, transition: Transition):
        """添加单步数据"""
        self._transitions.append(transition)
        
        # 管理 episode
        if self._current_episode is None:
            self._current_episode = Episode()
        
        self._current_episode.add(transition)
        
        if transition.done:
            self._episodes.append(self._current_episode)
            self._current_episode = None
    
    def add_episode(self, episode: Episode):
        """添加完整轨迹"""
        for transition in episode:
            self._transitions.append(transition)
        self._episodes.append(episode)
    
    def sample_transitions(self, batch_size: int) -> List[Transition]:
        """随机采样 transitions"""
        if len(self._transitions) < batch_size:
            return list(self._transitions)
        return random.sample(list(self._transitions), batch_size)
    
    def __len__(self) -> int:
        return len(self._transitions)
    
    @property
    def num_episodes(self) -> int:
        return len(self._episodes)
    
    def get_statistics(self) -> dict:
        return {
            "num_transitions": len(self),
            "num_episodes": self.num_episodes,
            "capacity": self.max_size,
            "utilization": len(self) / self.max_size,
        }
    
    def clear(self):
        self._transitions.clear()
        self._episodes.clear()
        self._current_episode = None
