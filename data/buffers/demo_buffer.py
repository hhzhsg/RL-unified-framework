"""专家演示数据 Buffer（离线采集的人类专家数据）"""
from typing import Dict, Any, List
from pathlib import Path
import numpy as np
from .base_buffer import BaseBuffer
from core.orchestration import register_buffer


@register_buffer("demo")
class DemoBuffer(BaseBuffer):
    """
    专家演示数据（只读，支持 HDF5/NPZ 格式）
    
    支持:
    - 单个文件: data_path 指向 .hdf5 / .npz 文件
    - 目录: data_path 指向包含 HDF5 文件的目录（递归加载）
    
    默认路径: data/demo/
    """
    
    def __init__(self, data_path: str = None, capacity: int = 0):
        super().__init__(capacity)
        self._data: Dict[str, List[np.ndarray]] = {}
        self._episodes: List[Dict[str, np.ndarray]] = []
        
        if data_path:
            self.load(data_path)
    
    def load(self, path: str) -> None:
        """
        加载数据
        
        Args:
            path: 文件路径或目录路径
        """
        path = Path(path)
        
        if path.is_dir():
            # 递归加载目录下所有 HDF5 文件
            hdf5_files = list(path.rglob("*.hdf5")) + list(path.rglob("*.h5"))
            if not hdf5_files:
                print(f"[DemoBuffer] Warning: No HDF5 files found in {path}")
                return
            
            print(f"[DemoBuffer] Loading {len(hdf5_files)} HDF5 files from {path}")
            for f in hdf5_files:
                self._load_hdf5(f)
        elif path.suffix == '.npz':
            self._load_npz(path)
        elif path.suffix in ['.hdf5', '.h5']:
            self._load_hdf5(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # 合并所有 episode 数据
        self._merge_data()
        print(f"[DemoBuffer] Loaded {self._size} transitions from {len(self._episodes)} episodes")
    
    def _load_hdf5(self, path: Path) -> None:
        """加载单个 HDF5 文件（支持嵌套 Group 结构）"""
        import h5py
        try:
            with h5py.File(path, 'r') as f:
                episode_data = {}
                
                def _extract_datasets(group, prefix=""):
                    """递归提取所有 Dataset"""
                    for key, item in group.items():
                        full_key = f"{prefix}/{key}" if prefix else key
                        if isinstance(item, h5py.Dataset):
                            episode_data[full_key] = np.array(item)
                        elif isinstance(item, h5py.Group):
                            _extract_datasets(item, full_key)
                
                _extract_datasets(f)
                if episode_data:
                    self._episodes.append(episode_data)
        except Exception as e:
            print(f"[DemoBuffer] Failed to load {path}: {e}")
    
    def _load_npz(self, path: Path) -> None:
        """加载 NPZ 文件"""
        data = np.load(path)
        episode_data = {k: data[k] for k in data.files}
        if episode_data:
            self._episodes.append(episode_data)
    
    def _merge_data(self) -> None:
        """合并所有 episode 数据"""
        if not self._episodes:
            self._size = 0
            self._capacity = 0
            return
        
        # 获取所有 keys
        all_keys = set()
        for ep in self._episodes:
            all_keys.update(ep.keys())
        
        # 合并
        self._data = {}
        for key in all_keys:
            arrays = [ep[key] for ep in self._episodes if key in ep]
            if arrays:
                self._data[key] = np.concatenate(arrays, axis=0)
        
        # 更新 size
        if self._data:
            first_key = list(self._data.keys())[0]
            self._size = len(self._data[first_key])
            self._capacity = self._size
    
    def add(self, data: Dict[str, Any]) -> None:
        raise RuntimeError("DemoBuffer is read-only")
    
    def add_batch(self, data: Dict[str, Any]) -> None:
        raise RuntimeError("DemoBuffer is read-only")
    
    def sample(self, batch_size: int) -> Dict[str, Any]:
        indices = np.random.randint(0, self._size, size=batch_size)
        return {k: v[indices] for k, v in self._data.items()}
