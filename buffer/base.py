"""
Buffer 基类

所有 Buffer 实现的抽象基类
"""
from abc import ABC, abstractmethod
from typing import List, Any
import pickle

from data import Transition, Episode


class BaseBuffer(ABC):
    """
    Buffer 基类
    
    所有 Buffer 实现需要继承此类并实现:
    - add_transition(): 添加单步数据
    - add_episode(): 添加完整轨迹
    - sample_transitions(): 采样 transitions
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
    def __len__(self) -> int:
        """返回 transition 数量"""
        pass
    
    @property
    def num_episodes(self) -> int:
        """Episode 数量 (子类可覆盖)"""
        return 0
    
    def save(self, path: str):
        """保存到文件"""
        with open(path, 'wb') as f:
            pickle.dump(self._get_save_data(), f)
    
    def load(self, path: str):
        """从文件加载"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self._load_from_data(data)
    
    def _get_save_data(self) -> Any:
        """获取需要保存的数据 (子类实现)"""
        return None
    
    def _load_from_data(self, data: Any):
        """从数据加载 (子类实现)"""
        pass
    
    def clear(self):
        """清空 buffer (子类实现)"""
        pass
    
    def statistics(self) -> dict:
        """获取统计信息"""
        return {
            "num_transitions": len(self),
            "num_episodes": self.num_episodes,
            "max_size": self.max_size,
        }
