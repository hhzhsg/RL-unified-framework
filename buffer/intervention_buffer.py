"""
VLA-RL Intervention Buffer
内存 + 异步落盘，用于 Human-in-the-Loop
"""
from __future__ import annotations
import os
import time
import threading
import queue
from typing import List, Optional, Dict, Any
from collections import deque
from datetime import datetime
import random
import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    import pickle  # 回退到 pickle

from .base_buffer import BaseBuffer
from data import Transition, Episode


class InterventionBuffer(BaseBuffer):
    """
    Intervention 缓冲区
    
    特点:
    - 内存中保存最近数据用于在线训练
    - 异步落盘，保存完整 episode 用于后续离线训练
    - 支持按任务/场景组织文件
    
    文件结构:
    save_dir/
        intervention_20240108_143052_task001.hdf5
        intervention_20240108_143521_task001.hdf5
        ...
    """
    
    def __init__(self, 
                 max_size: int = 50000,
                 save_dir: Optional[str] = None,
                 auto_save: bool = True,
                 save_interval: int = 100):
        """
        Args:
            max_size: 内存中最大 transition 数量
            save_dir: 落盘目录，None 则不落盘
            auto_save: 是否自动异步落盘
            save_interval: 每多少条 transition 触发一次落盘检查
        """
        super().__init__(max_size)
        
        self.save_dir = save_dir
        self.auto_save = auto_save
        self.save_interval = save_interval
        
        # 内存缓冲
        self._transitions: deque[Transition] = deque(maxlen=max_size)
        
        # 当前正在收集的 episode
        self._current_episode: List[Transition] = []
        self._current_task_id: str = ""
        
        # 已完成待落盘的 episodes
        self._pending_episodes: queue.Queue[Episode] = queue.Queue()
        
        # 落盘统计
        self._total_saved_episodes: int = 0
        self._total_saved_transitions: int = 0
        
        # 异步写入线程
        self._save_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        if save_dir and auto_save:
            self._start_save_thread()
    
    def _start_save_thread(self):
        """启动异步保存线程"""
        os.makedirs(self.save_dir, exist_ok=True)
        
        def save_worker():
            while not self._stop_event.is_set():
                try:
                    # 等待数据，超时 1 秒
                    episode = self._pending_episodes.get(timeout=1.0)
                    self._save_episode_to_disk(episode)
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"[InterventionBuffer] Save error: {e}")
        
        self._save_thread = threading.Thread(target=save_worker, daemon=True)
        self._save_thread.start()
    
    def _save_episode_to_disk(self, episode: Episode):
        """保存单个 episode 到磁盘 (HDF5 格式，与 Demo 数据一致)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = episode.task_id or "unknown"
        filename = f"intervention_{timestamp}_{task_id}.hdf5"
        filepath = os.path.join(self.save_dir, filename)
        
        if HAS_H5PY:
            self._save_episode_hdf5(episode, filepath)
        else:
            # 回退到 pickle
            import pickle
            pkl_path = filepath.replace('.hdf5', '.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump(episode, f)
            filepath = pkl_path
        
        self._total_saved_episodes += 1
        self._total_saved_transitions += len(episode)
        
        print(f"[InterventionBuffer] Saved episode to {filepath} "
              f"({len(episode)} transitions)")
    
    def _save_episode_hdf5(self, episode: Episode, filepath: str):
        """将 episode 保存为 HDF5 格式 (与 Demo 数据结构一致)"""
        transitions = episode.transitions
        num_steps = len(transitions)
        
        with h5py.File(filepath, 'w') as f:
            # 元信息
            f.attrs['task_id'] = episode.task_id or ""
            f.attrs['success'] = episode.success
            f.attrs['num_steps'] = num_steps
            f.attrs['timestamp'] = time.time()
            
            # 收集所有 state 和 action
            states = []
            actions = []
            rewards = []
            dones = []
            
            for t in transitions:
                states.append(t.robot_state.to_array())
                actions.append(t.action.data)
                rewards.append(t.reward)
                dones.append(t.done)
            
            states = np.stack(states)   # [T, state_dim]
            actions = np.stack(actions) # [T, action_dim]
            rewards = np.array(rewards) # [T]
            dones = np.array(dones)     # [T]
            
            # 保存 observation/state (与 Demo 格式一致)
            state_group = f.create_group('observation/state')
            # 保存完整 state 向量
            state_group.create_dataset('raw', data=states, compression='gzip')
            
            # 保存 action (与 Demo 格式一致)
            action_group = f.create_group('action')
            action_group.create_dataset('raw', data=actions, compression='gzip')
            
            # 保存 reward 和 done
            f.create_dataset('reward', data=rewards)
            f.create_dataset('done', data=dones)
            
            # 保存图像 (如果有)
            if transitions[0].obs.images:
                img_group = f.create_group('observation/images')
                for cam_key in transitions[0].obs.images.keys():
                    cam_group = img_group.create_group(cam_key)
                    # 收集所有帧的图像
                    images = []
                    for t in transitions:
                        if cam_key in t.obs.images:
                            images.append(t.obs.images[cam_key])
                    if images:
                        images = np.stack(images)  # [T, H, W, C]
                        cam_group.create_dataset('color', data=images, compression='gzip')
    
    def start_episode(self, task_id: str = ""):
        """开始收集新 episode"""
        # 如果有未完成的 episode，先结束它
        if self._current_episode:
            self.end_episode(success=False)
        
        self._current_episode = []
        self._current_task_id = task_id
    
    def add_transition(self, transition: Transition):
        """添加单步 intervention 数据"""
        transition.source = "intervention"
        
        # 添加到内存
        self._transitions.append(transition)
        
        # 添加到当前 episode
        self._current_episode.append(transition)
        
        # 如果 done，结束 episode
        if transition.done:
            self.end_episode(success=transition.reward > 0)
    
    def end_episode(self, success: bool = False):
        """结束当前 episode，触发落盘"""
        if not self._current_episode:
            return
        
        episode = Episode(
            transitions=self._current_episode.copy(),
            success=success,
            task_id=self._current_task_id,
            metadata={"timestamp": time.time()},
        )
        
        # 放入待保存队列
        if self.save_dir and self.auto_save:
            self._pending_episodes.put(episode)
        
        self._current_episode = []
    
    def add_episode(self, episode: Episode):
        """添加完整 episode"""
        for t in episode.transitions:
            t.source = "intervention"
            self._transitions.append(t)
        
        # 落盘
        if self.save_dir and self.auto_save:
            self._pending_episodes.put(episode)
    
    def sample_transitions(self, batch_size: int) -> List[Transition]:
        """从内存中随机采样"""
        if len(self._transitions) == 0:
            return []
        
        indices = [random.randint(0, len(self._transitions) - 1) 
                   for _ in range(batch_size)]
        return [self._transitions[i] for i in indices]
    
    def sample_episodes(self, batch_size: int) -> List[Episode]:
        """Intervention 通常按 transition 采样"""
        raise NotImplementedError("Use sample_transitions for intervention buffer")
    
    def load_from_disk(self, load_dir: Optional[str] = None):
        """从磁盘加载历史 intervention 数据 (支持 HDF5 和 pkl)"""
        load_dir = load_dir or self.save_dir
        if not load_dir or not os.path.exists(load_dir):
            return
        
        import glob
        from data import Observation, RobotState, Action
        
        # 优先加载 HDF5
        hdf5_files = glob.glob(os.path.join(load_dir, "intervention_*.hdf5"))
        pkl_files = glob.glob(os.path.join(load_dir, "intervention_*.pkl"))
        
        loaded_count = 0
        
        # 加载 HDF5 文件
        if HAS_H5PY:
            for filepath in hdf5_files:
                try:
                    with h5py.File(filepath, 'r') as f:
                        states = f['observation/state/raw'][:]
                        actions = f['action/raw'][:]
                        rewards = f['reward'][:]
                        dones = f['done'][:]
                        
                        for i in range(len(states)):
                            t = Transition(
                                obs=Observation(images={}),
                                robot_state=RobotState(joint_pos=states[i][:14], raw_state=states[i]),
                                action=Action(data=actions[i], space="joint"),
                                reward=float(rewards[i]),
                                next_obs=Observation(images={}),
                                next_robot_state=RobotState(
                                    joint_pos=states[min(i+1, len(states)-1)][:14],
                                    raw_state=states[min(i+1, len(states)-1)]
                                ),
                                done=bool(dones[i]),
                                source="intervention"
                            )
                            self._transitions.append(t)
                        loaded_count += 1
                except Exception as e:
                    print(f"[Warning] Failed to load {filepath}: {e}")
        
        # 加载 pkl 文件 (兼容旧格式)
        import pickle
        for filepath in pkl_files:
            try:
                with open(filepath, 'rb') as f:
                    episode = pickle.load(f)
                    for t in episode.transitions:
                        self._transitions.append(t)
                    loaded_count += 1
            except Exception as e:
                print(f"[Warning] Failed to load {filepath}: {e}")
        
        print(f"[InterventionBuffer] Loaded {loaded_count} episodes, "
              f"{len(self._transitions)} transitions from {load_dir}")
    
    def flush(self):
        """强制落盘当前 episode"""
        if self._current_episode:
            self.end_episode(success=False)
        
        # 等待队列清空
        while not self._pending_episodes.empty():
            time.sleep(0.1)
    
    def __len__(self) -> int:
        return len(self._transitions)
    
    @property
    def num_episodes(self) -> int:
        return self._total_saved_episodes
    
    def _get_save_data(self):
        """保存内存数据"""
        return list(self._transitions)
    
    def _load_from_data(self, data):
        """加载数据"""
        self._transitions = deque(data, maxlen=self.max_size)
    
    def clear(self):
        """清空内存数据 (不删除磁盘文件)"""
        self._transitions.clear()
        self._current_episode = []
    
    def get_statistics(self) -> dict:
        """获取统计信息"""
        return {
            "num_transitions_memory": len(self._transitions),
            "num_transitions_saved": self._total_saved_transitions,
            "num_episodes_saved": self._total_saved_episodes,
            "current_episode_length": len(self._current_episode),
            "pending_save": self._pending_episodes.qsize(),
            "save_dir": self.save_dir,
        }
    
    def __del__(self):
        """清理"""
        self._stop_event.set()
        if self._save_thread and self._save_thread.is_alive():
            self._save_thread.join(timeout=2.0)
