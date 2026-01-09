"""
VLA-RL 采样策略

支持从多个 buffer 按不同策略采样
"""
from abc import ABC, abstractmethod
from typing import List, Dict
import random

from data import Transition


class BaseSampleStrategy(ABC):
    """采样策略基类"""
    
    @abstractmethod
    def sample(self, buffers: Dict[str, "BaseBuffer"], batch_size: int) -> List[Transition]:
        """
        从多个 buffer 中采样
        
        Args:
            buffers: {"demo": buffer, "rollout": buffer, "intervention": buffer}
            batch_size: 采样数量
            
        Returns:
            transitions 列表
        """
        pass


class DemoOnlyStrategy(BaseSampleStrategy):
    """只从 demo buffer 采样"""
    
    def sample(self, buffers: Dict[str, "BaseBuffer"], batch_size: int) -> List[Transition]:
        demo_buffer = buffers.get("demo")
        if demo_buffer is None or len(demo_buffer) == 0:
            return []
        return demo_buffer.sample_transitions(batch_size)


class RolloutOnlyStrategy(BaseSampleStrategy):
    """只从 rollout buffer 采样"""
    
    def sample(self, buffers: Dict[str, "BaseBuffer"], batch_size: int) -> List[Transition]:
        rollout_buffer = buffers.get("rollout")
        if rollout_buffer is None or len(rollout_buffer) == 0:
            return []
        return rollout_buffer.sample_transitions(batch_size)


class MixedStrategy(BaseSampleStrategy):
    """混合采样"""
    
    def __init__(self, demo_ratio: float = 0.5, intervention_ratio: float = 0.0):
        """
        Args:
            demo_ratio: demo 数据占比
            intervention_ratio: intervention 数据占比
            rollout 占比 = 1 - demo_ratio - intervention_ratio
        """
        self.demo_ratio = demo_ratio
        self.intervention_ratio = intervention_ratio
        self.rollout_ratio = 1.0 - demo_ratio - intervention_ratio
        
        assert self.rollout_ratio >= 0, "Ratios must sum to <= 1.0"
    
    def sample(self, buffers: Dict[str, "BaseBuffer"], batch_size: int) -> List[Transition]:
        demo_buffer = buffers.get("demo")
        rollout_buffer = buffers.get("rollout")
        intervention_buffer = buffers.get("intervention")
        
        # 计算各部分采样数量
        demo_size = int(batch_size * self.demo_ratio)
        intervention_size = int(batch_size * self.intervention_ratio)
        rollout_size = batch_size - demo_size - intervention_size
        
        transitions = []
        
        # 采样 demo
        if demo_buffer and len(demo_buffer) > 0 and demo_size > 0:
            transitions.extend(demo_buffer.sample_transitions(demo_size))
        
        # 采样 intervention
        if intervention_buffer and len(intervention_buffer) > 0 and intervention_size > 0:
            transitions.extend(intervention_buffer.sample_transitions(intervention_size))
        
        # 采样 rollout
        if rollout_buffer and len(rollout_buffer) > 0 and rollout_size > 0:
            transitions.extend(rollout_buffer.sample_transitions(rollout_size))
        
        # 如果某个 buffer 为空，从其他 buffer 补充
        if len(transitions) < batch_size:
            all_buffers = [b for b in [demo_buffer, rollout_buffer, intervention_buffer] 
                         if b is not None and len(b) > 0]
            if all_buffers:
                need = batch_size - len(transitions)
                extra_buffer = random.choice(all_buffers)
                transitions.extend(extra_buffer.sample_transitions(need))
        
        return transitions


# 策略注册表
STRATEGY_REGISTRY = {
    "demo_only": DemoOnlyStrategy,
    "rollout_only": RolloutOnlyStrategy,
    "mixed": MixedStrategy,
}


def create_strategy(name: str, **kwargs) -> BaseSampleStrategy:
    """创建采样策略"""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY.keys())}")
    
    return STRATEGY_REGISTRY[name](**kwargs)
