"""
VLA-RL HDF5 Demo Buffer
支持大规模图像数据的 Lazy Loading
"""
from __future__ import annotations
import os
import glob
import random
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass
import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("[Warning] h5py not installed, HDF5 buffer will not work")

from .base_buffer import BaseBuffer
from data import Transition, Episode, Observation, RobotState, Action


@dataclass
class EpisodeIndex:
    """Episode 索引信息 (不加载实际数据)"""
    file_path: str
    episode_key: str  # HDF5 中的 key，如 "episode_0" 或根目录
    num_steps: int
    

class HDF5DemoBuffer(BaseBuffer):
    """
    HDF5 Demo Buffer
    
    特点:
    - 不预加载图像数据到内存
    - 采样时按需读取 (lazy loading)
    - 支持 JPEG 压缩的图像
    
    HDF5 结构 (基于你的数据):
    /images/
        cam_high: (T, H, W, 3) 或 JPEG bytes
        cam_left_wrist: (T, H, W, 3)
        cam_right_wrist: (T, H, W, 3)
    /observations/
        arm/position: (T, 14)
        arm/velocity: (T, 14)
        arm/torque: (T, 14)
        base/velocity: (T, 2)
        effector/pos: (T, 2)
        end/position: (T, 14)
        head/position: (T, 2)
        waist/position: (T, 3)
    /actions/
        arm/position: (T, 14)
        base/velocity: (T, 2)
        ...
    """
    
    # 相机名称
    CAMERA_KEYS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    
    # State 字段映射 (observation/state 下)
    STATE_KEYS = [
        ("arm/position", 14),
        ("arm/velocity", 14),
        ("arm/torque", 14),
        ("base/velocity", 2),
        ("effector/position", 2),
        ("end/position", 14),
        ("head/position", 2),
        ("waist/position", 3),
    ]  # 总共 65 维
    
    # Action 字段映射
    ACTION_KEYS = [
        ("arm/position", 14),
        ("base/velocity", 2),
        ("effector/position", 2),
        ("end/position", 14),
        ("head/position", 2),
        ("waist/position", 3),
    ]  # 总共 37 维
    
    def __init__(self, 
                 demo_paths: Optional[List[str]] = None,
                 max_size: int = 1000000,
                 camera_keys: Optional[List[str]] = None,
                 load_images: bool = True):
        """
        Args:
            demo_paths: HDF5 文件路径列表，支持 glob 模式
            max_size: 最大 transition 数量 (索引层面)
            camera_keys: 要加载的相机，None 表示全部
            load_images: 是否加载图像 (False 用于纯状态训练)
        """
        super().__init__(max_size)
        
        if not HAS_H5PY:
            raise ImportError("Please install h5py: pip install h5py")
        
        self.camera_keys = camera_keys or self.CAMERA_KEYS
        self.load_images = load_images
        
        # 索引: [(file_path, episode_key, step_idx), ...]
        self._indices: List[Tuple[str, str, int]] = []
        self._episode_indices: List[EpisodeIndex] = []
        
        # 文件句柄缓存 (避免频繁打开关闭)
        self._file_cache: Dict[str, h5py.File] = {}
        
        # 加载索引
        if demo_paths:
            self.load_demos(demo_paths)
    
    def load_demos(self, paths: List[str]):
        """加载 demo 文件索引 (不加载实际数据)"""
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
                # 检测文件结构 (你们的结构: observation/action 在根目录)
                if 'observation' in f or 'action' in f:
                    # 单 episode 文件
                    num_steps = self._get_num_steps(f)
                    ep_idx = EpisodeIndex(file_path, "", num_steps)
                    self._episode_indices.append(ep_idx)
                    
                    for step in range(num_steps):
                        self._indices.append((file_path, "", step))
                else:
                    # 多 episode 文件 (如果以后有)
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
    
    def _get_num_steps(self, group: h5py.Group) -> int:
        """获取 episode 步数"""
        # 尝试从 action 获取 (你们的结构)
        if 'action' in group:
            action_group = group['action']
            if 'arm' in action_group and 'position' in action_group['arm']:
                return action_group['arm/position'].shape[0]
        
        # 尝试从 observation 获取
        if 'observation' in group:
            obs_group = group['observation']
            if 'state' in obs_group:
                state_group = obs_group['state']
                if 'arm' in state_group and 'position' in state_group['arm']:
                    return state_group['arm/position'].shape[0]
        
        raise ValueError("Cannot determine episode length")
    
    def _get_file(self, file_path: str) -> h5py.File:
        """获取文件句柄 (带缓存)"""
        if file_path not in self._file_cache:
            self._file_cache[file_path] = h5py.File(file_path, 'r')
        return self._file_cache[file_path]
    
    def _get_group(self, file_path: str, episode_key: str) -> h5py.Group:
        """获取 episode 数据组"""
        f = self._get_file(file_path)
        if episode_key:
            return f[episode_key]
        return f
    
    def _load_images(self, group: h5py.Group, step: int) -> Dict[str, np.ndarray]:
        """加载单步图像"""
        images = {}
        if not self.load_images:
            return images
        
        # 你们的结构: observation/images/{cam_key}/color
        if 'observation' not in group or 'images' not in group['observation']:
            return images
        
        img_group = group['observation/images']
        for cam_key in self.camera_keys:
            if cam_key in img_group and 'color' in img_group[cam_key]:
                img_data = img_group[cam_key]['color'][step]
                
                # 处理 JPEG bytes (object dtype)
                if isinstance(img_data, (bytes, np.bytes_)):
                    import cv2
                    img_array = np.frombuffer(img_data, dtype=np.uint8)
                    img_data = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
                
                images[cam_key] = img_data.astype(np.float32) / 255.0
        
        return images
    
    def _load_state(self, group: h5py.Group, step: int) -> np.ndarray:
        """加载单步状态"""
        # 你们的结构: observation/state/{key}
        state_group = group['observation/state']
        parts = []
        
        for key, dim in self.STATE_KEYS:
            if key in state_group:
                parts.append(state_group[key][step])
            else:
                # 字段不存在，填零
                parts.append(np.zeros(dim, dtype=np.float64))
        
        return np.concatenate(parts).astype(np.float32)
    
    def _load_action(self, group: h5py.Group, step: int) -> np.ndarray:
        """加载单步动作"""
        # 你们的结构: action/{key}
        action_group = group['action']
        parts = []
        
        for key, dim in self.ACTION_KEYS:
            if key in action_group:
                parts.append(action_group[key][step])
            else:
                parts.append(np.zeros(dim, dtype=np.float64))
        
        return np.concatenate(parts).astype(np.float32)
    
    def _load_transition(self, file_path: str, episode_key: str, step: int) -> Transition:
        """加载单个 transition"""
        group = self._get_group(file_path, episode_key)
        num_steps = self._get_num_steps(group)
        
        # 当前步
        images = self._load_images(group, step)
        state = self._load_state(group, step)
        action = self._load_action(group, step)
        
        # 下一步 (最后一步用自己)
        next_step = min(step + 1, num_steps - 1)
        next_images = self._load_images(group, next_step)
        next_state = self._load_state(group, next_step)
        
        # 构造 RobotState
        robot_state = self._array_to_robot_state(state)
        next_robot_state = self._array_to_robot_state(next_state)
        
        # done 和 reward
        done = (step == num_steps - 1)
        reward = 1.0 if done else 0.0  # 稀疏奖励
        
        return Transition(
            obs=Observation(images=images),
            robot_state=robot_state,
            action=Action(data=action, space="joint"),
            reward=reward,
            next_obs=Observation(images=next_images),
            next_robot_state=next_robot_state,
            done=done,
            source="demo",
        )
    
    def _array_to_robot_state(self, state: np.ndarray) -> RobotState:
        """将拼接的状态向量转回 RobotState"""
        # 直接保存原始 65 维向量，避免信息丢失
        # 按 STATE_KEYS 顺序解析字段位置:
        # arm/position: 0:14, arm/velocity: 14:28, arm/torque: 28:42
        # base/velocity: 42:44, effector/pos: 44:46, end/position: 46:60
        # head/position: 60:62, waist/position: 62:65
        
        return RobotState(
            joint_pos=state[0:14],         # arm/position
            joint_vel=state[14:28],        # arm/velocity
            ee_pos=state[46:52] if len(state) > 52 else np.zeros(3, dtype=np.float32),  # end/position 前3维
            gripper=float(state[44]) if len(state) > 44 else 0.0,  # effector/pos[0]
            raw_state=state,               # 保存完整 65 维
        )
    
    # ========== BaseBuffer 接口实现 ==========
    
    def add_transition(self, transition: Transition):
        """Demo buffer 通常只读，但保留接口"""
        raise NotImplementedError("HDF5DemoBuffer is read-only")
    
    def add_episode(self, episode: Episode):
        """Demo buffer 通常只读"""
        raise NotImplementedError("HDF5DemoBuffer is read-only")
    
    def sample_transitions(self, batch_size: int) -> List[Transition]:
        """随机采样 transitions"""
        if len(self._indices) == 0:
            return []
        
        sampled_indices = random.choices(self._indices, k=batch_size)
        transitions = []
        
        for file_path, episode_key, step in sampled_indices:
            try:
                t = self._load_transition(file_path, episode_key, step)
                transitions.append(t)
            except Exception as e:
                print(f"[Warning] Failed to load transition: {e}")
                continue
        
        return transitions
    
    def sample_episodes(self, batch_size: int) -> List[Episode]:
        """采样完整 episodes (较慢，谨慎使用)"""
        if len(self._episode_indices) == 0:
            return []
        
        sampled = random.choices(self._episode_indices, k=batch_size)
        episodes = []
        
        for ep_idx in sampled:
            try:
                transitions = []
                for step in range(ep_idx.num_steps):
                    t = self._load_transition(ep_idx.file_path, ep_idx.episode_key, step)
                    transitions.append(t)
                
                ep = Episode(
                    transitions=transitions,
                    success=True,  # Demo 默认成功
                    task_id=os.path.basename(ep_idx.file_path),
                )
                episodes.append(ep)
            except Exception as e:
                print(f"[Warning] Failed to load episode: {e}")
                continue
        
        return episodes
    
    def __len__(self) -> int:
        return len(self._indices)
    
    @property
    def num_episodes(self) -> int:
        return len(self._episode_indices)
    
    def _get_save_data(self):
        """不支持保存 (只读)"""
        raise NotImplementedError("HDF5DemoBuffer does not support save")
    
    def _load_from_data(self, data):
        """不支持这种加载方式"""
        raise NotImplementedError("Use load_demos() instead")
    
    def clear(self):
        """清空索引和缓存"""
        self._indices.clear()
        self._episode_indices.clear()
        for f in self._file_cache.values():
            f.close()
        self._file_cache.clear()
    
    def get_statistics(self) -> dict:
        """获取统计信息"""
        return {
            "num_episodes": self.num_episodes,
            "num_transitions": len(self),
            "num_files": len(set(idx[0] for idx in self._indices)),
            "camera_keys": self.camera_keys,
            "state_dim": sum(dim for _, dim in self.STATE_KEYS),
            "action_dim": sum(dim for _, dim in self.ACTION_KEYS),
        }
    
    def __del__(self):
        """清理文件句柄"""
        for f in self._file_cache.values():
            try:
                f.close()
            except:
                pass


# ============ 工具函数 ============

def inspect_hdf5(file_path: str):
    """检查 HDF5 文件结构"""
    import h5py
    
    def print_attrs(name, obj):
        indent = "  " * name.count('/')
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}{name}: {obj.shape} {obj.dtype}")
        else:
            print(f"{indent}{name}/")
    
    with h5py.File(file_path, 'r') as f:
        print(f"=== {file_path} ===")
        f.visititems(print_attrs)
