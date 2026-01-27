"""Replay Buffer - 支持持久化的 Rollout 数据存储"""
from typing import Dict, Any, List, Optional
from pathlib import Path
import threading
import queue
import pickle
import time
import glob
import numpy as np
from .base_buffer import BaseBuffer
from core.orchestration import register_buffer


@register_buffer("replay")
class ReplayBuffer(BaseBuffer):
    """
    Rollout 数据 Buffer
    
    对标 HIL-SERL 的 data_store / replay_buffer:
    - 环形缓冲: 内存中固定容量
    - 定期持久化: 按步数周期保存到 Pickle
    - 可恢复: 下次训练可加载历史数据
    - 线程安全: insert/sample 加锁
    """
    
    def __init__(
        self, 
        capacity: int,
        save_path: Optional[str] = None,
        save_interval: int = 1000,  # 每 N 步保存一次
        obs_shape: Dict[str, tuple] = None,
    ):
        """
        Args:
            capacity: 环形缓冲容量
            save_path: 持久化目录 (如 checkpoint_path/buffer/)
            save_interval: 每多少步触发一次保存
            obs_shape: (已废弃，保留兼容)
        """
        super().__init__(capacity)
        self._storage: List[Dict[str, Any]] = []
        self._pos = 0
        self._lock = threading.Lock()
        
        # 持久化相关
        self._save_path = Path(save_path) if save_path else None
        self._save_interval = save_interval
        self._step_count = 0
        self._last_save_step = 0
        self._unsaved_data: List[Dict[str, Any]] = []
        
        # 异步保存
        self._save_queue: queue.Queue = queue.Queue()
        self._save_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        if self._save_path:
            self._save_path.mkdir(parents=True, exist_ok=True)
            self._start_save_thread()
    
    def _start_save_thread(self):
        """启动后台保存线程"""
        self._save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self._save_thread.start()
    
    def _save_worker(self):
        """后台保存线程"""
        while not self._stop_event.is_set():
            try:
                data, step = self._save_queue.get(timeout=1.0)
                self._do_save(data, step)
            except queue.Empty:
                continue
    
    def _do_save(self, data: List[Dict[str, Any]], step: int):
        """执行保存"""
        if not self._save_path or not data:
            return
        filename = self._save_path / f"transitions_{step}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"[ReplayBuffer] Saved {len(data)} transitions at step {step}")
    
    def add(self, data: Dict[str, Any]) -> None:
        """添加单条 transition"""
        with self._lock:
            if len(self._storage) < self._capacity:
                self._storage.append(data)
            else:
                self._storage[self._pos] = data
            self._pos = (self._pos + 1) % self._capacity
            self._size = min(self._size + 1, self._capacity)
            
            # 持久化追踪
            self._step_count += 1
            self._unsaved_data.append(data.copy())
            
            # 触发保存
            if self._save_path and (self._step_count - self._last_save_step) >= self._save_interval:
                self._trigger_save()
    
    def insert(self, data: Dict[str, Any]) -> None:
        """HIL-SERL 风格的别名"""
        self.add(data)
    
    def _trigger_save(self):
        """触发异步保存"""
        if self._unsaved_data:
            self._save_queue.put((self._unsaved_data.copy(), self._step_count))
            self._unsaved_data = []
            self._last_save_step = self._step_count
    
    def add_batch(self, data: Dict[str, Any]) -> None:
        """批量添加"""
        batch_size = len(data[list(data.keys())[0]])
        for i in range(batch_size):
            item = {k: v[i] for k, v in data.items()}
            self.add(item)
    
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """随机采样"""
        with self._lock:
            if self._size == 0:
                return {}
            indices = np.random.randint(0, self._size, size=min(batch_size, self._size))
            batch = [self._storage[i] for i in indices]
        
        result = {}
        for key in batch[0].keys():
            values = [b[key] for b in batch]
            if isinstance(values[0], np.ndarray):
                result[key] = np.stack(values)
            elif isinstance(values[0], dict):
                # 嵌套 dict (如 observations)
                result[key] = {}
                for subkey in values[0].keys():
                    subvals = [v[subkey] for v in values]
                    if isinstance(subvals[0], np.ndarray):
                        result[key][subkey] = np.stack(subvals)
                    else:
                        result[key][subkey] = np.array(subvals)
            else:
                result[key] = np.array(values)
        return result
    
    def load(self, path: Optional[str] = None) -> int:
        """
        加载历史数据
        
        Args:
            path: 数据目录，默认使用 save_path
            
        Returns:
            加载的 transition 数量
        """
        load_path = Path(path) if path else self._save_path
        if not load_path or not load_path.exists():
            return 0
        
        loaded_count = 0
        # 按文件名排序加载
        pkl_files = sorted(glob.glob(str(load_path / "transitions_*.pkl")))
        for pkl_file in pkl_files:
            with open(pkl_file, 'rb') as f:
                transitions = pickle.load(f)
                for t in transitions:
                    # 直接添加，不触发保存
                    with self._lock:
                        if len(self._storage) < self._capacity:
                            self._storage.append(t)
                        else:
                            self._storage[self._pos] = t
                        self._pos = (self._pos + 1) % self._capacity
                        self._size = min(self._size + 1, self._capacity)
                    loaded_count += 1
        
        print(f"[ReplayBuffer] Loaded {loaded_count} transitions from {len(pkl_files)} files")
        return loaded_count
    
    def flush(self):
        """强制保存未保存的数据"""
        if self._unsaved_data:
            self._trigger_save()
        # 等待队列清空
        start = time.time()
        while not self._save_queue.empty() and time.time() - start < 5.0:
            time.sleep(0.1)
    
    def close(self):
        """关闭 buffer"""
        self.flush()
        self._stop_event.set()
        if self._save_thread and self._save_thread.is_alive():
            self._save_thread.join(timeout=5.0)
    
    def clear(self) -> None:
        """清空缓冲"""
        with self._lock:
            self._storage = []
            self._pos = 0
            self._size = 0
    
    def __del__(self):
        try:
            self.close()
        except:
            pass
    
    @property
    def step_count(self) -> int:
        """总步数"""
        return self._step_count
