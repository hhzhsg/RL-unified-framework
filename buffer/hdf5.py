"""
HDF5 Demo Buffer

支持大规模图像数据的 Lazy Loading
"""
from __future__ import annotations
import os
import glob
import random
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass
import numpy as np

from .base import BaseBuffer
from data import Transition, Episode, Observation, RobotState, Action

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


@dataclass
class EpisodeIndex:
    """Episode 索引信息 (不加载实际数据)"""
    file_path: str
    episode_key: str
    num_steps: int


class HDF5DemoBuffer(BaseBuffer):
    """
    HDF5 Demo Buffer
    
    特点:
    - 不预加载图像数据到内存
    - 采样时按需读取 (lazy loading)
    - 支持 JPEG 压缩的图像
    
    Example:
        buffer = HDF5DemoBuffer(demo_paths=["data/*.hdf5"])
        transitions = buffer.sample_transitions(64)
    """
    
    # 默认相机名称
    DEFAULT_CAMERA_KEYS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    
    def __init__(self, 
                 demo_paths: Optional[List[str]] = None,
                 max_size: int = 1000000,
                 camera_keys: Optional[List[str]] = None,
                 load_images: bool = True):
        """
        Args:
            demo_paths: HDF5 文件路径列表，支持 glob 模式
            max_size: 最大 transition 数量 (索引层面)
            camera_keys: 要加载的相机列表
            load_images: 是否加载图像
        """
        super().__init__(max_size)
        
        if not HAS_H5PY:
            raise ImportError("Please install h5py: pip install h5py")
        
        self.camera_keys = camera_keys or self.DEFAULT_CAMERA_KEYS
        self.load_images = load_images
        
        # 索引: [(file_path, episode_key, step_idx), ...]
        self._indices: List[Tuple[str, str, int]] = []
        self._episode_indices: List[EpisodeIndex] = []
        
        # 文件句柄缓存
        self._file_cache: Dict[str, h5py.File] = {}
        
        if demo_paths:
            self.load_demos(demo_paths)
    
    def load_demos(self, paths: List[str]):
        """加载 demo 文件索引"""
        all_files = []
        for path in paths:
            if '*' in path:
                all_files.extend(glob.glob(path))
            elif os.path.isdir(path):
                all_files.extend(glob.glob(os.path.join(path, "*.hdf5")))
            elif os.path.isfile(path):
                all_files.append(path)
        
        for file_path in all_files:
            self._index_file(file_path)
        
        print(f"[HDF5DemoBuffer] Loaded {len(self._episode_indices)} episodes, "
              f"{len(self._indices)} transitions from {len(all_files)} files")
    
    def _index_file(self, file_path: str):
        """索引单个 HDF5 文件"""
        try:
            with h5py.File(file_path, 'r') as f:
                # 检测文件结构
                if 'observation' in f or 'action' in f:
                    # 单 episode 文件
                    num_steps = self._get_num_steps(f)
                    ep_idx = EpisodeIndex(file_path, "", num_steps)
                    self._episode_indices.append(ep_idx)
                    
                    for step in range(num_steps):
                        self._indices.append((file_path, "", step))
                else:
                    # 多 episode 文件
                    for key in f.keys():
                        if key.startswith('episode') or key.startswith('demo'):
                            ep_group = f[key]
                            num_steps = self._get_num_steps(ep_group)
                            ep_idx = EpisodeIndex(file_path, key, num_steps)
                            self._episode_indices.append(ep_idx)
                            
                            for step in range(num_steps):
                                self._indices.append((file_path, key, step))
        except Exception as e:
            print(f"[Warning] Failed to index {file_path}: {e}")
    
    def _get_num_steps(self, group) -> int:
        """获取 episode 步数"""
        # 尝试不同路径
        for path in ['action', 'actions', 'observation/state']:
            if path in group:
                data = group[path]
                if hasattr(data, 'shape'):
                    return data.shape[0]
                # 可能是嵌套结构
                for subkey in data.keys():
                    return data[subkey].shape[0]
        return 0
    
    def _get_file(self, path: str) -> h5py.File:
        """获取文件句柄 (带缓存)"""
        if path not in self._file_cache:
            self._file_cache[path] = h5py.File(path, 'r')
        return self._file_cache[path]
    
    def _load_transition(self, file_path: str, ep_key: str, step: int) -> Transition:
        """加载单个 transition"""
        f = self._get_file(file_path)
        group = f[ep_key] if ep_key else f
        
        # 加载状态
        state = self._load_state(group, step)
        next_state = self._load_state(group, min(step + 1, self._get_num_steps(group) - 1))
        
        # 加载动作
        action = self._load_action(group, step)
        
        # 加载图像 (可选)
        images = {}
        if self.load_images:
            images = self._load_images(group, step)
        
        return Transition(
            obs=Observation(images=images),
            robot_state=RobotState(raw_state=state),
            action=Action(data=action),
            reward=0.0,  # demo 没有 reward
            next_obs=Observation(images={}),
            next_robot_state=RobotState(raw_state=next_state),
            done=(step == self._get_num_steps(group) - 1),
            source="demo",
        )
    
    def _load_state(self, group, step: int) -> np.ndarray:
        """加载状态"""
        # 尝试不同路径
        for base in ['observation/state', 'observations']:
            if base in group:
                state_group = group[base]
                parts = []
                for key in state_group.keys():
                    data = state_group[key][step]
                    if isinstance(data, np.ndarray):
                        parts.append(data.flatten())
                    else:
                        parts.append(np.array([data]))
                if parts:
                    return np.concatenate(parts).astype(np.float32)
        return np.zeros(16, dtype=np.float32)
    
    def _load_action(self, group, step: int) -> np.ndarray:
        """加载动作"""
        for path in ['action', 'actions']:
            if path in group:
                action_data = group[path]
                if hasattr(action_data, 'shape') and len(action_data.shape) >= 1:
                    return np.array(action_data[step], dtype=np.float32)
                # 嵌套结构
                parts = []
                for key in action_data.keys():
                    parts.append(action_data[key][step].flatten())
                return np.concatenate(parts).astype(np.float32)
        return np.zeros(7, dtype=np.float32)
    
    def _load_images(self, group, step: int) -> Dict[str, np.ndarray]:
        """加载图像"""
        images = {}
        for base in ['images', 'observation/images']:
            if base in group:
                img_group = group[base]
                for cam in self.camera_keys:
                    if cam in img_group:
                        img_data = img_group[cam][step]
                        # 处理 JPEG 压缩
                        if img_data.dtype == np.uint8 and len(img_data.shape) == 1:
                            try:
                                import cv2
                                img_data = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                            except ImportError:
                                continue
                        images[cam] = img_data
                break
        return images
    
    def add_transition(self, transition: Transition):
        """Demo buffer 是只读的"""
        raise NotImplementedError("HDF5DemoBuffer is read-only")
    
    def add_episode(self, episode: Episode):
        """Demo buffer 是只读的"""
        raise NotImplementedError("HDF5DemoBuffer is read-only")
    
    def sample_transitions(self, batch_size: int) -> List[Transition]:
        """随机采样"""
        if not self._indices:
            return []
        
        batch_size = min(batch_size, len(self._indices))
        sampled = random.sample(self._indices, batch_size)
        
        return [self._load_transition(*idx) for idx in sampled]
    
    def __len__(self) -> int:
        return len(self._indices)
    
    @property
    def num_episodes(self) -> int:
        return len(self._episode_indices)
    
    def close(self):
        """关闭所有文件句柄"""
        for f in self._file_cache.values():
            f.close()
        self._file_cache.clear()
    
    def __del__(self):
        self.close()


def inspect_hdf5(path: str) -> Dict[str, Any]:
    """
    检查 HDF5 文件结构
    
    Args:
        path: HDF5 文件路径
        
    Returns:
        文件结构信息
    """
    if not HAS_H5PY:
        raise ImportError("Please install h5py")
    
    def _inspect_group(group, prefix=""):
        info = {}
        for key in group.keys():
            item = group[key]
            full_key = f"{prefix}/{key}" if prefix else key
            if hasattr(item, 'shape'):
                info[full_key] = {
                    "shape": item.shape,
                    "dtype": str(item.dtype),
                }
            else:
                info.update(_inspect_group(item, full_key))
        return info
    
    with h5py.File(path, 'r') as f:
        return _inspect_group(f)
