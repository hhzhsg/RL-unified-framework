"""
Intervention Buffer

存储 rollout 过程中人类专家介入的数据，支持持久化
"""
import os
import pickle
from typing import List, Optional

from .base import BaseBuffer
from data import Transition, Episode


class InterventionBuffer(BaseBuffer):
    """
    Intervention Buffer
    
    用于存储在线 rollout 过程中人类专家介入纠正的高质量数据。
    
    特点:
    - 数据来源: rollout 中人工接管/纠正的轨迹片段
    - 持久化存储 (不会因为重启丢失)
    - FIFO 替换策略
    - 支持保存/加载
    
    与其他 buffer 的区别:
    - demo: 预训练的离线专家数据 (固定，只读)
    - rollout: policy 自主探索的轨迹 (质量不稳定)
    - intervention: 人工介入的高质量数据 (持久化，可复用)
    
    Example:
        buffer = InterventionBuffer(max_size=10000, save_path="./data/intervention.pkl")
        
        # 添加干预数据
        buffer.add_transition(transition)
        
        # 自动保存
        buffer.save()
        
        # 下次启动时加载
        buffer.load()
    """
    
    def __init__(self, max_size: int = 10000, save_path: Optional[str] = None):
        """
        Args:
            max_size: 最大容量
            save_path: 持久化文件路径
        """
        super().__init__(max_size)
        self.save_path = save_path
        self._transitions: List[Transition] = []
        self._episode_count = 0
        
        # 如果指定了路径且文件存在，自动加载
        if save_path and os.path.exists(save_path):
            self.load()
    
    def add_transition(self, transition: Transition):
        """添加单步数据"""
        self._transitions.append(transition)
        
        # FIFO 替换
        if len(self._transitions) > self.max_size:
            self._transitions.pop(0)
        
        # 自动保存
        if self.save_path:
            self.save()
    
    def add_episode(self, episode: Episode):
        """添加完整轨迹"""
        for t in episode.transitions:
            self._transitions.append(t)
        
        # FIFO 替换
        while len(self._transitions) > self.max_size:
            self._transitions.pop(0)
        
        self._episode_count += 1
        
        # 自动保存
        if self.save_path:
            self.save()
    
    def sample_transitions(self, batch_size: int) -> List[Transition]:
        """随机采样"""
        if len(self._transitions) == 0:
            return []
        
        import random
        batch_size = min(batch_size, len(self._transitions))
        return random.sample(self._transitions, batch_size)
    
    def __len__(self) -> int:
        return len(self._transitions)
    
    @property
    def num_episodes(self) -> int:
        return self._episode_count
    
    def clear(self):
        """清空数据"""
        self._transitions.clear()
        self._episode_count = 0
        if self.save_path:
            self.save()
    
    def save(self, path: Optional[str] = None):
        """
        保存到文件
        
        Args:
            path: 保存路径 (默认使用 self.save_path)
        """
        save_path = path or self.save_path
        if not save_path:
            return
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        data = {
            "transitions": self._transitions,
            "episode_count": self._episode_count,
            "max_size": self.max_size,
        }
        
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
    
    def load(self, path: Optional[str] = None):
        """
        从文件加载
        
        Args:
            path: 加载路径 (默认使用 self.save_path)
        """
        load_path = path or self.save_path
        if not load_path or not os.path.exists(load_path):
            return
        
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        
        self._transitions = data["transitions"]
        self._episode_count = data.get("episode_count", 0)
        self.max_size = data.get("max_size", self.max_size)
    
    def _get_save_data(self):
        """用于 BaseBuffer 的保存接口"""
        return {
            "transitions": self._transitions,
            "episode_count": self._episode_count,
        }
    
    def _load_from_data(self, data):
        """用于 BaseBuffer 的加载接口"""
        self._transitions = data["transitions"]
        self._episode_count = data.get("episode_count", 0)
