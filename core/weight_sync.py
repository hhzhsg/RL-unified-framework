"""
VLA-RL 权重同步
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
from queue import Queue, Empty
import threading


class BaseWeightSync(ABC):
    """权重同步基类"""
    
    @abstractmethod
    def push(self, state_dict: Dict[str, Any], version: int):
        """推送权重"""
        pass
    
    @abstractmethod
    def pull(self) -> Optional[Tuple[Dict[str, Any], int]]:
        """拉取最新权重，返回 (state_dict, version) 或 None"""
        pass


class QueueWeightSync(BaseWeightSync):
    """
    基于 Queue 的权重同步
    适用于单机多进程
    """
    
    def __init__(self, maxsize: int = 2):
        self.queue = Queue(maxsize=maxsize)
        self._latest_version = 0
    
    def push(self, state_dict: Dict[str, Any], version: int):
        """推送权重 (非阻塞，队列满则丢弃旧的)"""
        try:
            # 清空旧的
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except Empty:
                    break
            
            self.queue.put_nowait({"state_dict": state_dict, "version": version})
            self._latest_version = version
        except:
            pass  # 队列满，跳过
    
    def pull(self) -> Optional[Tuple[Dict[str, Any], int]]:
        """拉取最新权重 (非阻塞)"""
        try:
            data = self.queue.get_nowait()
            return data["state_dict"], data["version"]
        except Empty:
            return None


class SharedMemoryWeightSync(BaseWeightSync):
    """
    基于共享内存的权重同步
    适用于高频同步场景
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._state_dict = None
        self._version = 0
    
    def push(self, state_dict: Dict[str, Any], version: int):
        """推送权重"""
        with self._lock:
            self._state_dict = state_dict
            self._version = version
    
    def pull(self) -> Optional[Tuple[Dict[str, Any], int]]:
        """拉取最新权重"""
        with self._lock:
            if self._state_dict is None:
                return None
            return self._state_dict, self._version


# 注册表
WEIGHT_SYNC_REGISTRY = {
    "queue": QueueWeightSync,
    "shared_memory": SharedMemoryWeightSync,
}


def create_weight_sync(method: str = "queue", **kwargs) -> BaseWeightSync:
    """创建权重同步器"""
    if method not in WEIGHT_SYNC_REGISTRY:
        raise ValueError(f"Unknown method: {method}. Available: {list(WEIGHT_SYNC_REGISTRY.keys())}")
    
    return WEIGHT_SYNC_REGISTRY[method](**kwargs)
