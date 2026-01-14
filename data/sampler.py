"""
采样策略

定义不同的数据采样方式:
- DemoOnlySampler: 仅从演示数据采样
- RolloutOnlySampler: 仅从在线数据采样
- MixedSampler: 混合采样
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import random

from .types import Transition


class BaseSampler(ABC):
    """采样策略基类"""
    
    @abstractmethod
    def sample(self, buffers: Dict[str, "BaseBuffer"], batch_size: int) -> List[Transition]:
        """
        从多个 buffer 中采样
        
        Args:
            buffers: {name: buffer} 字典
            batch_size: 采样数量
            
        Returns:
            Transition 列表
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称"""
        pass


class DemoOnlySampler(BaseSampler):
    """仅从演示数据采样"""
    
    @property
    def name(self) -> str:
        return "demo_only"
    
    def sample(self, buffers: Dict[str, "BaseBuffer"], batch_size: int) -> List[Transition]:
        demo_buffer = buffers.get("demo")
        if demo_buffer is None or len(demo_buffer) == 0:
            raise ValueError("DemoOnlySampler requires non-empty 'demo' buffer")
        return demo_buffer.sample_transitions(batch_size)


class RolloutOnlySampler(BaseSampler):
    """仅从在线数据采样"""
    
    @property
    def name(self) -> str:
        return "rollout_only"
    
    def sample(self, buffers: Dict[str, "BaseBuffer"], batch_size: int) -> List[Transition]:
        rollout_buffer = buffers.get("rollout")
        if rollout_buffer is None or len(rollout_buffer) == 0:
            raise ValueError("RolloutOnlySampler requires non-empty 'rollout' buffer")
        return rollout_buffer.sample_transitions(batch_size)


class MixedSampler(BaseSampler):
    """
    混合采样
    
    按比例从 demo 和 rollout 采样，当 rollout 为空时自动退化为纯 demo
    """
    
    def __init__(self, demo_ratio: float = 0.25):
        """
        Args:
            demo_ratio: demo 数据占比 (0~1)
        """
        self.demo_ratio = demo_ratio
    
    @property
    def name(self) -> str:
        return f"mixed_{self.demo_ratio:.2f}"
    
    def sample(self, buffers: Dict[str, "BaseBuffer"], batch_size: int) -> List[Transition]:
        demo_buffer = buffers.get("demo")
        rollout_buffer = buffers.get("rollout")
        
        has_demo = demo_buffer is not None and len(demo_buffer) > 0
        has_rollout = rollout_buffer is not None and len(rollout_buffer) > 0
        
        # 自动退化逻辑
        if not has_demo and not has_rollout:
            raise ValueError("MixedSampler requires at least one non-empty buffer")
        if not has_rollout:
            return demo_buffer.sample_transitions(batch_size)
        if not has_demo:
            return rollout_buffer.sample_transitions(batch_size)
        
        # 混合采样
        demo_size = int(batch_size * self.demo_ratio)
        rollout_size = batch_size - demo_size
        
        transitions = []
        transitions.extend(demo_buffer.sample_transitions(demo_size))
        transitions.extend(rollout_buffer.sample_transitions(rollout_size))
        random.shuffle(transitions)
        
        return transitions


# 采样器注册表
SAMPLER_REGISTRY = {
    "demo_only": DemoOnlySampler,
    "rollout_only": RolloutOnlySampler,
    "mixed": MixedSampler,
}


def create_sampler(name: str, **kwargs) -> BaseSampler:
    """
    创建采样器
    
    Args:
        name: 采样器名称
        **kwargs: 采样器参数
        
    Returns:
        采样器实例
    """
    if name not in SAMPLER_REGISTRY:
        raise ValueError(f"Unknown sampler: {name}. Available: {list(SAMPLER_REGISTRY.keys())}")
    return SAMPLER_REGISTRY[name](**kwargs)
