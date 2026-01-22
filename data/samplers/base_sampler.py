"""采样器基类"""
from abc import abstractmethod
from typing import Dict, Any
from core.interfaces import SamplerInterface


class BaseSampler(SamplerInterface):
    """采样器基类"""
    
    @abstractmethod
    def sample(self, buffers: Dict[str, Any], batch_size: int) -> Dict[str, Any]:
        pass
    
    @property
    def name(self) -> str:
        return self.__class__.__name__
