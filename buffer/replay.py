"""
Replay Buffer

通用经验回放缓冲区，支持 FIFO 替换策略
"""
from typing import List, Any, Optional
from collections import deque
import random

from .base import BaseBuffer
from data import Transition, Episode


class ReplayBuffer(BaseBuffer):
    """
    Replay Buffer
    
    FIFO 替换策略的经验回放缓冲区
    
    Example:
        buffer = ReplayBuffer(max_size=10000)
        buffer.add_transition(transition)
        batch = buffer.sample_transitions(64)
    """
    
    def __init__(self, max_size: int = 100000):
        super().__init__(max_size)
        self._transitions: deque = deque(maxlen=max_size)
        self._episode_count = 0
    
    def add_transition(self, transition: Transition):
        """添加单步数据"""
        self._transitions.append(transition)
    
    def add_episode(self, episode: Episode):
        """添加完整轨迹"""
        for t in episode.transitions:
            self._transitions.append(t)
        self._episode_count += 1
    
    def sample_transitions(self, batch_size: int) -> List[Transition]:
        """随机采样"""
        if len(self._transitions) == 0:
            return []
        
        batch_size = min(batch_size, len(self._transitions))
        return random.sample(list(self._transitions), batch_size)
    
    def __len__(self) -> int:
        return len(self._transitions)
    
    @property
    def num_episodes(self) -> int:
        return self._episode_count
    
    def clear(self):
        """清空"""
        self._transitions.clear()
        self._episode_count = 0
    
    def _get_save_data(self) -> Any:
        return {
            "transitions": list(self._transitions),
            "episode_count": self._episode_count,
        }
    
    def _load_from_data(self, data: Any):
        self._transitions = deque(data["transitions"], maxlen=self.max_size)
        self._episode_count = data.get("episode_count", 0)


class SimpleReplayBuffer(ReplayBuffer):
    """SimpleReplayBuffer 的别名 (向后兼容)"""
    pass


class RolloutBuffer(ReplayBuffer):
    """RolloutBuffer 的别名 (向后兼容)"""
    pass
