"""
权重发布器

训练端使用，发布最新权重
"""
from typing import Dict, Any, Optional
import torch

from .base_sync import BaseSynchronizer


class WeightPublisher:
    """
    权重发布器
    
    封装同步器，提供权重发布接口
    """
    
    def __init__(self, synchronizer: BaseSynchronizer, tag: str = "weights"):
        self.synchronizer = synchronizer
        self.tag = tag
        self._publish_count = 0
    
    def publish(self, state_dict: Dict[str, torch.Tensor], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        发布权重
        
        Args:
            state_dict: 模型状态字典
            metadata: 元数据（如训练步数等）
        """
        # 将tensor转为CPU并序列化
        cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
        
        data = {
            "state_dict": cpu_state_dict,
            "metadata": metadata or {},
            "version": self._publish_count,
        }
        
        self.synchronizer.push(data, tag=self.tag)
        self._publish_count += 1
    
    @property
    def publish_count(self) -> int:
        return self._publish_count
