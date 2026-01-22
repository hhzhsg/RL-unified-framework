"""
队列同步器

适用于多进程场景
"""
from typing import Dict, Any, Optional
from queue import Queue, Empty
import copy

from ..base_sync import BaseSynchronizer


class QueueSync(BaseSynchronizer):
    """
    队列同步器
    
    通过Queue传递数据（适用于多进程）
    """
    
    def __init__(self, maxsize: int = 1):
        self._queues: Dict[str, Queue] = {}
        self._versions: Dict[str, int] = {}
        self._maxsize = maxsize
    
    def _get_queue(self, tag: str) -> Queue:
        """获取或创建队列"""
        if tag not in self._queues:
            self._queues[tag] = Queue(maxsize=self._maxsize)
            self._versions[tag] = 0
        return self._queues[tag]
    
    def push(self, data: Dict[str, Any], tag: str = "default") -> None:
        """推送数据（非阻塞，丢弃旧数据）"""
        queue = self._get_queue(tag)
        
        # 清空旧数据
        while not queue.empty():
            try:
                queue.get_nowait()
            except Empty:
                break
        
        queue.put(copy.deepcopy(data))
        self._versions[tag] = self._versions.get(tag, 0) + 1
    
    def pull(self, tag: str = "default") -> Optional[Dict[str, Any]]:
        """拉取数据（非阻塞）"""
        queue = self._get_queue(tag)
        try:
            return queue.get_nowait()
        except Empty:
            return None
    
    def get_version(self, tag: str = "default") -> int:
        """获取版本号"""
        return self._versions.get(tag, 0)
