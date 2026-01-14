"""
权重同步

训练进程和推理进程之间的权重同步机制
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from queue import Queue, Empty
import copy


class BaseWeightSync(ABC):
    """权重同步基类"""
    
    @abstractmethod
    def push(self, state_dict: Dict[str, Any], version: int = 0):
        """推送权重"""
        pass
    
    @abstractmethod
    def pull(self) -> Optional[Tuple[Dict[str, Any], int]]:
        """拉取权重，返回 (state_dict, version) 或 None"""
        pass
    
    @abstractmethod
    def get_version(self) -> int:
        """获取当前版本"""
        pass


class SharedMemorySync(BaseWeightSync):
    """
    共享内存同步
    
    简单实现: 直接共享 Python 对象
    适用于单机多线程
    """
    
    def __init__(self):
        self._state_dict: Optional[Dict[str, Any]] = None
        self._version: int = 0
    
    def push(self, state_dict: Dict[str, Any], version: int = 0):
        """推送权重"""
        # 深拷贝避免引用问题
        self._state_dict = copy.deepcopy(state_dict)
        self._version = version
    
    def pull(self) -> Optional[Tuple[Dict[str, Any], int]]:
        """拉取权重"""
        if self._state_dict is None:
            return None
        return copy.deepcopy(self._state_dict), self._version
    
    def get_version(self) -> int:
        return self._version


class QueueSync(BaseWeightSync):
    """
    队列同步
    
    使用 Queue 传递权重
    适用于多线程
    """
    
    def __init__(self, maxsize: int = 1):
        self._queue: Queue = Queue(maxsize=maxsize)
        self._version: int = 0
    
    def push(self, state_dict: Dict[str, Any], version: int = 0):
        """推送权重 (非阻塞，丢弃旧的)"""
        # 清空队列
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break
        
        self._queue.put((copy.deepcopy(state_dict), version))
        self._version = version
    
    def pull(self) -> Optional[Tuple[Dict[str, Any], int]]:
        """拉取权重 (非阻塞)"""
        try:
            return self._queue.get_nowait()
        except Empty:
            return None
    
    def get_version(self) -> int:
        return self._version


# 同步器注册表
SYNC_REGISTRY = {
    "shared_memory": SharedMemorySync,
    "queue": QueueSync,
}


def create_weight_sync(method: str = "shared_memory") -> BaseWeightSync:
    """
    创建权重同步器
    
    Args:
        method: 同步方法 ("shared_memory" | "queue")
        
    Returns:
        同步器实例
    """
    if method not in SYNC_REGISTRY:
        raise ValueError(f"Unknown sync method: {method}. Available: {list(SYNC_REGISTRY.keys())}")
    return SYNC_REGISTRY[method]()
