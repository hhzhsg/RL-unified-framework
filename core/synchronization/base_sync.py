"""
同步器基类
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseSynchronizer(ABC):
    """同步器基类"""
    
    @abstractmethod
    def push(self, data: Dict[str, Any], tag: str = "default") -> None:
        """
        推送数据
        
        Args:
            data: 数据字典
            tag: 数据标签
        """
        pass
    
    @abstractmethod
    def pull(self, tag: str = "default") -> Optional[Dict[str, Any]]:
        """
        拉取数据
        
        Args:
            tag: 数据标签
            
        Returns:
            数据字典，如果没有新数据返回None
        """
        pass
    
    @abstractmethod
    def get_version(self, tag: str = "default") -> int:
        """获取版本号"""
        pass
    
    def close(self) -> None:
        """关闭同步器"""
        pass
