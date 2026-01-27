"""Intervention Buffer - 人工介入数据存储

对标 HIL-SERL 的 intvn_data_store / demo_buffer:
- 只存储带 intervention 标记的 transition
- 支持持久化和恢复
- 与 ReplayBuffer 共享相同的数据格式
"""
from typing import Dict, Any, Optional, List
from pathlib import Path
import threading
import queue
import pickle
import time
import glob
import numpy as np
from .replay_buffer import ReplayBuffer
from core.orchestration import register_buffer


@register_buffer("intervention")
class InterventionBuffer(ReplayBuffer):
    """
    Intervention 数据 Buffer
    
    对标 HIL-SERL 的 intvn_data_store / demo_buffer:
    - 继承 ReplayBuffer 的环形缓冲和持久化能力
    - 只接收 intervention 数据（通过 env.info["intervene_action"] 判断）
    - 保存到单独目录 (checkpoint_path/demo_buffer/)
    - Learner 端用于混合采样
    
    与 ReplayBuffer 的区别:
    - 文件名使用 demo_buffer 前缀而非 transitions
    - 自动添加 source="intervention" 标记
    """
    
    def __init__(
        self, 
        capacity: int,
        save_path: Optional[str] = None,
        save_interval: int = 100,  # intervention 通常更少，间隔小一些
        async_save: bool = True,
    ):
        """
        Args:
            capacity: 内存缓冲容量
            save_path: 持久化目录 (如 checkpoint_path/demo_buffer/)
            save_interval: 每多少条触发一次保存
            async_save: 是否异步保存
        """
        # 不调用父类 __init__ 以避免重复启动线程
        # 手动初始化
        from .base_buffer import BaseBuffer
        BaseBuffer.__init__(self, capacity)
        
        self._storage: List[Dict[str, Any]] = []
        self._pos = 0
        self._lock = threading.Lock()
        
        self._save_path = Path(save_path) if save_path else None
        self._save_interval = save_interval
        self._async_save = async_save
        self._step_count = 0
        self._last_save_step = 0
        self._unsaved_data: List[Dict[str, Any]] = []
        self._total_saved = 0
        
        # 异步保存
        self._save_queue: queue.Queue = queue.Queue()
        self._save_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        if self._save_path:
            self._save_path.mkdir(parents=True, exist_ok=True)
            if self._async_save:
                self._start_save_thread()
    
    def _do_save(self, data: List[Dict[str, Any]], step: int):
        """执行保存 - 使用 demo_buffer 命名"""
        if not self._save_path or not data:
            return
        filename = self._save_path / f"transitions_{step}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        self._total_saved += len(data)
        print(f"[InterventionBuffer] Saved {len(data)} interventions at step {step}, total saved: {self._total_saved}")
    
    def add(self, data: Dict[str, Any]) -> None:
        """
        添加 intervention transition
        
        自动添加:
        - source: "intervention"
        - timestamp: 当前时间
        """
        data = data.copy()
        data["source"] = "intervention"
        if "timestamp" not in data:
            data["timestamp"] = time.time()
        
        with self._lock:
            if len(self._storage) < self._capacity:
                self._storage.append(data)
            else:
                self._storage[self._pos] = data
            self._pos = (self._pos + 1) % self._capacity
            self._size = min(self._size + 1, self._capacity)
            
            self._step_count += 1
            self._unsaved_data.append(data.copy())
            
            if self._save_path and (self._step_count - self._last_save_step) >= self._save_interval:
                self._trigger_save()
    
    def load(self, path: Optional[str] = None) -> int:
        """
        加载历史 intervention 数据
        
        支持两种文件格式:
        - transitions_*.pkl (新格式)
        - intervention_*.pkl (旧格式)
        """
        load_path = Path(path) if path else self._save_path
        if not load_path or not load_path.exists():
            return 0
        
        loaded_count = 0
        
        # 加载新格式
        pkl_files = sorted(glob.glob(str(load_path / "transitions_*.pkl")))
        # 加载旧格式
        pkl_files += sorted(glob.glob(str(load_path / "intervention_*.pkl")))
        
        for pkl_file in pkl_files:
            with open(pkl_file, 'rb') as f:
                transitions = pickle.load(f)
                for t in transitions:
                    with self._lock:
                        if len(self._storage) < self._capacity:
                            self._storage.append(t)
                        else:
                            self._storage[self._pos] = t
                        self._pos = (self._pos + 1) % self._capacity
                        self._size = min(self._size + 1, self._capacity)
                    loaded_count += 1
        
        print(f"[InterventionBuffer] Loaded {loaded_count} interventions from {len(pkl_files)} files")
        return loaded_count
    
    @property
    def total_saved(self) -> int:
        """已持久化的数据总数"""
        return self._total_saved
    
    def save_all(self, path: Optional[str] = None):
        """
        保存当前所有数据到指定路径
        """
        save_path = Path(path) if path else self._save_path
        if not save_path:
            raise ValueError("No save path specified")
        
        save_path.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time() * 1000)
        filename = save_path / f"intervention_full_{timestamp}.pkl"
        
        with self._lock:
            all_data = [self._storage[i] for i in range(len(self._storage))]
        
        with open(filename, 'wb') as f:
            pickle.dump(all_data, f)
        
        print(f"[InterventionBuffer] Saved all {len(all_data)} interventions to {filename}")
