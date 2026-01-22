"""
Sampler接口定义

定义如何从Buffer中采样数据
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List


class SamplerInterface(ABC):
    """采样器接口"""
    
    @abstractmethod
    def sample(self, buffers: Dict[str, "BufferInterface"], batch_size: int) -> Dict[str, Any]:
        """
        从多个buffer中采样数据
        
        Args:
            buffers: buffer字典 {name: buffer}
            batch_size: 批量大小
            
        Returns:
            采样的批量数据
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """采样器名称"""
        pass


class WeightedSamplerInterface(SamplerInterface):
    """加权采样器接口"""
    
    @property
    @abstractmethod
    def weights(self) -> Dict[str, float]:
        """各buffer的采样权重"""
        pass
    
    @abstractmethod
    def set_weights(self, weights: Dict[str, float]) -> None:
        """设置采样权重"""
        pass
