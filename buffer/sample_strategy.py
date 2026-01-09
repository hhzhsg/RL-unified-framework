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
    """
    混合采样策略
    
    支持 Demo + Rollout + Intervention 按比例混合采样
    关键特性：当 Rollout 为空时，自动退化为纯 Demo 采样
    """
    
    def __init__(self, demo_ratio: float = 0.25, intervention_ratio: float = 0.0):
        """
        Args:
            demo_ratio: demo 数据目标占比 (默认 25%)
            intervention_ratio: intervention 数据目标占比
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
        
        # 检查各 buffer 可用性
        demo_available = demo_buffer is not None and len(demo_buffer) > 0
        rollout_available = rollout_buffer is not None and len(rollout_buffer) > 0
        intervention_available = intervention_buffer is not None and len(intervention_buffer) > 0
        
        # ========== 关键逻辑：Rollout 为空时退化为纯 Demo ==========
        if not rollout_available and not intervention_available:
            # 只有 Demo 可用 → 纯 Demo 采样
            if demo_available:
                return demo_buffer.sample_transitions(batch_size)
            else:
                return []
        
        if not rollout_available:
            # Rollout 为空，Demo 和 Intervention 按比例分配
            if demo_available and intervention_available:
                total_ratio = self.demo_ratio + self.intervention_ratio
                demo_size = int(batch_size * self.demo_ratio / total_ratio)
                intervention_size = batch_size - demo_size
                transitions = demo_buffer.sample_transitions(demo_size)
                transitions.extend(intervention_buffer.sample_transitions(intervention_size))
                return transitions
            elif demo_available:
                return demo_buffer.sample_transitions(batch_size)
            elif intervention_available:
                return intervention_buffer.sample_transitions(batch_size)
            else:
                return []
        
        # ========== 正常情况：按目标比例采样 ==========
        transitions = []
        
        # 计算目标采样数量
        demo_size = int(batch_size * self.demo_ratio) if demo_available else 0
        intervention_size = int(batch_size * self.intervention_ratio) if intervention_available else 0
        rollout_size = batch_size - demo_size - intervention_size
        
        # 采样
        if demo_size > 0:
            transitions.extend(demo_buffer.sample_transitions(demo_size))
        if intervention_size > 0:
            transitions.extend(intervention_buffer.sample_transitions(intervention_size))
        if rollout_size > 0:
            transitions.extend(rollout_buffer.sample_transitions(rollout_size))
        
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
