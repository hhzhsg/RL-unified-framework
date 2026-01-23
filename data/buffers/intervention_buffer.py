"""干预数据Buffer - 支持异步落盘和加载"""
from typing import Dict, Any, Optional, List
from pathlib import Path
import threading
import queue
import pickle
import time
import numpy as np
from .replay_buffer import ReplayBuffer
from core.orchestration import register_buffer


@register_buffer("intervention")
class InterventionBuffer(ReplayBuffer):
    """
    干预数据专用Buffer
    
    特性:
    - 继承 ReplayBuffer 的内存环形缓冲
    - 异步落盘: 后台线程定期保存到磁盘
    - 可加载历史数据: 下次训练可加载之前场景的 intervention 数据
    """
    
    def __init__(
        self, 
        capacity: int,
        save_path: Optional[str] = None,
        save_interval: int = 100,  # 每 N 条数据触发一次保存
        async_save: bool = True,
    ):
        """
        Args:
            capacity: 内存缓冲容量
            save_path: 持久化目录路径
            save_interval: 每插入多少条数据触发一次异步保存
            async_save: 是否启用异步保存
        """
        super().__init__(capacity)
        self._save_path = Path(save_path) if save_path else None
        self._save_interval = save_interval
        self._async_save = async_save
        self._unsaved_count = 0
        self._total_saved = 0
        
        # 异步保存队列和线程
        self._save_queue: queue.Queue = queue.Queue()
        self._save_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        if self._save_path and self._async_save:
            self._start_save_thread()
    
    def _start_save_thread(self):
        """启动后台保存线程"""
        self._save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self._save_thread.start()
    
    def _save_worker(self):
        """后台保存线程工作函数"""
        while not self._stop_event.is_set():
            try:
                # 等待保存任务，超时 1 秒检查停止信号
                data_batch = self._save_queue.get(timeout=1.0)
                self._do_save(data_batch)
            except queue.Empty:
                continue
    
    def _do_save(self, data_batch: List[Dict[str, Any]]):
        """执行实际保存操作"""
        if not self._save_path or not data_batch:
            return
        
        self._save_path.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time() * 1000)
        filename = self._save_path / f"intervention_{timestamp}_{len(data_batch)}.pkl"
        
        with open(filename, 'wb') as f:
            pickle.dump(data_batch, f)
        
        self._total_saved += len(data_batch)
    
    def add(self, data: Dict[str, Any]) -> None:
        """添加数据并标记来源"""
        data = data.copy()
        data["source"] = "intervention"
        data["timestamp"] = time.time()
        super().add(data)
        
        self._unsaved_count += 1
        
        # 触发异步保存
        if self._save_path and self._unsaved_count >= self._save_interval:
            self._trigger_save()
    
    def _trigger_save(self):
        """触发异步保存"""
        # 收集待保存数据
        to_save = []
        start_idx = max(0, self._size - self._unsaved_count)
        for i in range(start_idx, self._size):
            idx = (self._pos - self._size + i) % self._capacity
            if idx >= 0 and idx < len(self._storage):
                to_save.append(self._storage[idx].copy())
        
        if to_save:
            if self._async_save:
                self._save_queue.put(to_save)
            else:
                self._do_save(to_save)
        
        self._unsaved_count = 0
    
    def flush(self):
        """强制保存所有未保存的数据"""
        if self._unsaved_count > 0:
            self._trigger_save()
        
        # 等待队列清空（带超时）
        if self._async_save and self._save_queue:
            # 等待最多 5 秒
            start = time.time()
            while not self._save_queue.empty() and time.time() - start < 5.0:
                time.sleep(0.1)
    
    def load(self, path: Optional[str] = None) -> int:
        """
        加载历史 intervention 数据
        
        Args:
            path: 数据目录路径，默认使用 save_path
            
        Returns:
            加载的数据条数
        """
        load_path = Path(path) if path else self._save_path
        if not load_path or not load_path.exists():
            return 0
        
        loaded_count = 0
        for pkl_file in sorted(load_path.glob("intervention_*.pkl")):
            with open(pkl_file, 'rb') as f:
                data_batch = pickle.load(f)
                for item in data_batch:
                    super().add(item)
                    loaded_count += 1
        
        return loaded_count
    
    def save_all(self, path: Optional[str] = None):
        """
        保存当前所有数据到指定路径
        
        Args:
            path: 保存路径，默认使用 save_path
        """
        save_path = Path(path) if path else self._save_path
        if not save_path:
            raise ValueError("No save path specified")
        
        save_path.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time() * 1000)
        filename = save_path / f"intervention_full_{timestamp}.pkl"
        
        all_data = [self._storage[i] for i in range(len(self._storage))]
        with open(filename, 'wb') as f:
            pickle.dump(all_data, f)
    
    def close(self):
        """关闭 buffer，保存剩余数据"""
        self.flush()
        self._stop_event.set()
        if self._save_thread and self._save_thread.is_alive():
            self._save_thread.join(timeout=5.0)
    
    def __del__(self):
        self.close()
    
    @property
    def total_saved(self) -> int:
        """已持久化的数据总数"""
        return self._total_saved
