"""
数据中心

统一管理多个数据源 (demo, rollout, intervention)，提供统一的采样接口
"""
from typing import Dict, List, Optional, Union
import torch

from .types import Transition, Batch
from .sampler import BaseSampler, create_sampler


class DataHub:
    """
    数据中心
    
    职责:
    1. 管理多个 Buffer (demo, rollout, intervention)
    2. 提供统一的写入/采样接口
    3. 支持不同采样策略
    
    Example:
        hub = DataHub()
        hub.register_buffer("demo", demo_buffer)
        hub.register_buffer("rollout", rollout_buffer)
        
        # 写入数据
        hub.write(transition, source="rollout")
        
        # 采样
        batch = hub.sample(batch_size=64, strategy="mixed")
    """
    
    def __init__(self, rollout_capacity: int = 100000):
        """
        Args:
            rollout_capacity: rollout buffer 容量 (便捷参数)
        """
        self._buffers: Dict[str, "BaseBuffer"] = {}
        self._samplers: Dict[str, BaseSampler] = {}
        self._default_sampler = "demo_only"
        
        # 便捷初始化: 自动创建 rollout buffer
        from buffer import ReplayBuffer
        self._buffers["rollout"] = ReplayBuffer(max_size=rollout_capacity)
    
    def register_buffer(self, name: str, buffer: "BaseBuffer"):
        """
        注册 buffer
        
        Args:
            name: buffer 名称 ("demo" | "rollout" | "intervention")
            buffer: buffer 实例
        """
        self._buffers[name] = buffer
    
    def get_buffer(self, name: str) -> Optional["BaseBuffer"]:
        """获取指定 buffer"""
        return self._buffers.get(name)
    
    @property
    def demo_buffer(self) -> Optional["BaseBuffer"]:
        return self._buffers.get("demo")
    
    @property
    def rollout_buffer(self) -> "BaseBuffer":
        return self._buffers["rollout"]
    
    def write(self, data: Union[Transition, "Episode"], source: str = "rollout"):
        """
        写入数据
        
        Args:
            data: Transition 或 Episode
            source: 目标 buffer 名称
        """
        buffer = self._buffers.get(source)
        if buffer is None:
            raise ValueError(f"Buffer '{source}' not registered")
        
        from .types import Episode
        if isinstance(data, Episode):
            buffer.add_episode(data)
        else:
            buffer.add_transition(data)
    
    def sample(self, batch_size: int, strategy: str = "demo_only", **kwargs) -> Batch:
        """
        采样数据
        
        Args:
            batch_size: 采样数量
            strategy: 采样策略 ("demo_only" | "rollout_only" | "mixed")
            **kwargs: 传递给采样器的参数
            
        Returns:
            训练 Batch
        """
        # 获取或创建采样器
        if strategy not in self._samplers:
            self._samplers[strategy] = create_sampler(strategy, **kwargs)
        
        sampler = self._samplers[strategy]
        transitions = sampler.sample(self._buffers, batch_size)
        
        return Batch.from_transitions(transitions)
    
    def statistics(self) -> Dict[str, int]:
        """获取各 buffer 统计信息"""
        stats = {}
        for name, buffer in self._buffers.items():
            stats[f"{name}_size"] = len(buffer)
            if hasattr(buffer, "num_episodes"):
                stats[f"{name}_episodes"] = buffer.num_episodes
        return stats
    
    def __repr__(self) -> str:
        stats = self.statistics()
        parts = [f"{k}={v}" for k, v in stats.items()]
        return f"DataHub({', '.join(parts)})"
