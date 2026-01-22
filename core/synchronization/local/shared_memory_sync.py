"""
共享内存同步器

适用于单机多线程场景
"""
from typing import Dict, Any, Optional
import copy
import threading

from ..base_sync import BaseSynchronizer


class SharedMemorySync(BaseSynchronizer):
    """
    共享内存同步器
    
    通过Python对象直接共享（适用于多线程）
    """
    
    def __init__(self):
        self._data: Dict[str, Dict[str, Any]] = {}
        self._versions: Dict[str, int] = {}
        self._lock = threading.Lock()
    
    def push(self, data: Dict[str, Any], tag: str = "default") -> None:
        """推送数据"""
        with self._lock:
            self._data[tag] = copy.deepcopy(data)
            self._versions[tag] = self._versions.get(tag, 0) + 1
    
    def pull(self, tag: str = "default") -> Optional[Dict[str, Any]]:
        """拉取数据"""
        with self._lock:
            if tag not in self._data:
                return None
            return copy.deepcopy(self._data[tag])
    
    def get_version(self, tag: str = "default") -> int:
        """获取版本号"""
        with self._lock:
            return self._versions.get(tag, 0)
