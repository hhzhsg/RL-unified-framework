"""Replay Buffer"""
from typing import Dict, Any, List
import numpy as np
from .base_buffer import BaseBuffer
from core.orchestration import register_buffer


@register_buffer("replay")
class ReplayBuffer(BaseBuffer):
    """标准Replay Buffer"""
    
    def __init__(self, capacity: int, obs_shape: Dict[str, tuple] = None):
        super().__init__(capacity)
        self._storage: List[Dict[str, Any]] = []
        self._pos = 0
    
    def add(self, data: Dict[str, Any]) -> None:
        if len(self._storage) < self._capacity:
            self._storage.append(data)
        else:
            self._storage[self._pos] = data
        self._pos = (self._pos + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)
    
    def add_batch(self, data: Dict[str, Any]) -> None:
        batch_size = len(data[list(data.keys())[0]])
        for i in range(batch_size):
            item = {k: v[i] for k, v in data.items()}
            self.add(item)
    
    def sample(self, batch_size: int) -> Dict[str, Any]:
        indices = np.random.randint(0, self._size, size=batch_size)
        batch = [self._storage[i] for i in indices]
        
        result = {}
        for key in batch[0].keys():
            values = [b[key] for b in batch]
            if isinstance(values[0], np.ndarray):
                result[key] = np.stack(values)
            else:
                result[key] = np.array(values)
        return result
    
    def clear(self) -> None:
        self._storage = []
        self._pos = 0
        self._size = 0
