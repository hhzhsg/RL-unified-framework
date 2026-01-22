"""Buffer基类"""
from abc import abstractmethod
from typing import Dict, Any
from core.interfaces import BufferInterface


class BaseBuffer(BufferInterface):
    """Buffer基类"""
    
    def __init__(self, capacity: int):
        self._capacity = capacity
        self._size = 0
    
    @property
    def capacity(self) -> int:
        return self._capacity
    
    def __len__(self) -> int:
        return self._size
    
    @abstractmethod
    def add(self, data: Dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    def add_batch(self, data: Dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    def sample(self, batch_size: int) -> Dict[str, Any]:
        pass
