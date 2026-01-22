"""离线数据集"""
from typing import Dict, Any
import numpy as np
from .base_buffer import BaseBuffer
from core.orchestration import register_buffer


@register_buffer("offline")
class OfflineDataset(BaseBuffer):
    """离线数据集（只读）"""
    
    def __init__(self, data_path: str = None, capacity: int = 0):
        super().__init__(capacity)
        self._data: Dict[str, np.ndarray] = {}
        if data_path:
            self.load(data_path)
    
    def load(self, path: str) -> None:
        """加载数据"""
        import os
        if path.endswith('.npz'):
            data = np.load(path)
            self._data = {k: data[k] for k in data.files}
        elif path.endswith('.hdf5') or path.endswith('.h5'):
            import h5py
            with h5py.File(path, 'r') as f:
                self._data = {k: np.array(f[k]) for k in f.keys()}
        self._size = len(self._data[list(self._data.keys())[0]])
        self._capacity = self._size
    
    def add(self, data: Dict[str, Any]) -> None:
        raise RuntimeError("OfflineDataset is read-only")
    
    def add_batch(self, data: Dict[str, Any]) -> None:
        raise RuntimeError("OfflineDataset is read-only")
    
    def sample(self, batch_size: int) -> Dict[str, Any]:
        indices = np.random.randint(0, self._size, size=batch_size)
        return {k: v[indices] for k, v in self._data.items()}
