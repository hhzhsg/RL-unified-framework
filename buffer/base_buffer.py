"""
VLA-RL Buffer 基类
"""
from abc import ABC, abstractmethod
from typing import List, Optional
import pickle

from data import Transition, Episode


class BaseBuffer(ABC):
    """
    Buffer 基类
    """
    
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
    
    @abstractmethod
    def add_transition(self, transition: Transition):
        """添加单步数据"""
        pass
    
    @abstractmethod
    def add_episode(self, episode: Episode):
        """添加完整轨迹"""
        pass
    
    @abstractmethod
    def sample_transitions(self, batch_size: int) -> List[Transition]:
        """采样 transitions"""
        pass
    
    @abstractmethod
    def sample_episodes(self, batch_size: int) -> List[Episode]:
        """采样 episodes"""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """返回数据量 (transition 数量)"""
        pass
    
    @property
    @abstractmethod
    def num_episodes(self) -> int:
        """Episode 数量"""
        pass
    
    def save(self, path: str):
        """保存到文件"""
        with open(path, 'wb') as f:
            pickle.dump(self._get_save_data(), f)
    
    def load(self, path: str):
        """从文件加载"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self._load_from_data(data)
    
    @abstractmethod
    def _get_save_data(self):
        """获取需要保存的数据"""
        pass
    
    @abstractmethod
    def _load_from_data(self, data):
        """从数据加载"""
        pass
    
    def clear(self):
        """清空 buffer"""
        pass
