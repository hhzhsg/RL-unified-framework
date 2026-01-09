"""
简化版 ReplayBuffer (向后兼容)
用于 pkl 格式的 demo 数据
"""
from typing import List
import random

from .base_buffer import BaseBuffer
from data import Transition, Episode


class SimpleReplayBuffer(BaseBuffer):
    """简化的 ReplayBuffer，用于 pkl demo 数据"""
    
    def __init__(self, max_size: int = 1000000):
        super().__init__(max_size)
        self.episodes: List[Episode] = []
        self._total_transitions = 0
    
    def add_transition(self, transition: Transition):
        """添加单步"""
        episode = Episode(transitions=[transition], success=False, task_id="")
        self.add_episode(episode)
    
    def add_episode(self, episode: Episode):
        """添加 episode"""
        self.episodes.append(episode)
        self._total_transitions += len(episode)
    
    def sample_transitions(self, batch_size: int) -> List[Transition]:
        """采样"""
        if len(self.episodes) == 0:
            return []
        
        transitions = []
        for _ in range(batch_size):
            ep = random.choice(self.episodes)
            if len(ep) > 0:
                t = random.choice(ep.transitions)
                transitions.append(t)
        return transitions
    
    def sample_episodes(self, batch_size: int) -> List[Episode]:
        """采样 episodes"""
        return random.choices(self.episodes, k=min(batch_size, len(self.episodes)))
    
    def __len__(self) -> int:
        return self._total_transitions
    
    @property
    def num_episodes(self) -> int:
        return len(self.episodes)
    
    def _get_save_data(self):
        return self.episodes
    
    def _load_from_data(self, data):
        self.episodes = data
        self._total_transitions = sum(len(ep) for ep in self.episodes)
    
    def clear(self):
        self.episodes = []
        self._total_transitions = 0
    
    def get_statistics(self) -> dict:
        if len(self.episodes) == 0:
            return {"num_episodes": 0, "num_transitions": 0}
        
        success_count = sum(1 for ep in self.episodes if ep.success)
        return {
            "num_episodes": len(self.episodes),
            "num_transitions": self._total_transitions,
            "success_rate": success_count / len(self.episodes) if len(self.episodes) > 0 else 0,
        }
