"""
VLA-RL WeightSync - 权重同步机制

用于训练进程和推理进程之间同步模型权重
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from queue import Queue, Empty
import copy


class BaseWeightSync(ABC):
    """
    权重同步基类
    
    训练端: push(state_dict, version)
    推理端: pull() -> (state_dict, version) or None
    """
    
    @abstractmethod
    def push(self, state_dict: Dict[str, Any], version: int):
        """训练端推送权重"""
        pass
    
    @abstractmethod
    def pull(self) -> Optional[Tuple[Dict[str, Any], int]]:
        """推理端拉取权重"""
        pass


class SharedMemorySync(BaseWeightSync):
    """
    共享内存同步 (单进程多线程)
    
    使用 Python 对象直接共享，适用于线程间同步
    """
    
    def __init__(self):
        self._state_dict: Optional[Dict[str, Any]] = None
        self._version: int = 0
        self._consumed: bool = True
    
    def push(self, state_dict: Dict[str, Any], version: int):
        """推送权重 (深拷贝)"""
        self._state_dict = copy.deepcopy(state_dict)
        self._version = version
        self._consumed = False
    
    def pull(self) -> Optional[Tuple[Dict[str, Any], int]]:
        """拉取权重"""
        if self._consumed or self._state_dict is None:
            return None
        
        self._consumed = True
        return copy.deepcopy(self._state_dict), self._version


class QueueSync(BaseWeightSync):
    """
    队列同步 (多进程)
    
    使用 Queue 传递权重，适用于进程间同步
    """
    
    def __init__(self, maxsize: int = 2):
        self._queue: Queue = Queue(maxsize=maxsize)
    
    def push(self, state_dict: Dict[str, Any], version: int):
        """推送权重"""
        # 清空旧数据
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break
        
        self._queue.put((copy.deepcopy(state_dict), version))
    
    def pull(self) -> Optional[Tuple[Dict[str, Any], int]]:
        """拉取权重"""
        try:
            return self._queue.get_nowait()
        except Empty:
            return None


# 注册表
WEIGHT_SYNC_REGISTRY = {
    "shared_memory": SharedMemorySync,
    "queue": QueueSync,
}


def create_weight_sync(method: str = "shared_memory") -> BaseWeightSync:
    """创建权重同步器"""
    if method not in WEIGHT_SYNC_REGISTRY:
        raise ValueError(f"未知同步方式: {method}。可用: {list(WEIGHT_SYNC_REGISTRY.keys())}")
    return WEIGHT_SYNC_REGISTRY[method]()
