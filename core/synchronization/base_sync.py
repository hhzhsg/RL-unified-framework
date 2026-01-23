"""
同步器基类

定义同步器的抽象接口
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseSynchronizer(ABC):
    """
    同步器基类
    
    用于训练端和推理端之间的数据同步
    """
    
    @abstractmethod
    def push(self, data: Dict[str, Any], tag: str = "default") -> None:
        """
        推送数据
        
        Args:
            data: 要推送的数据
            tag: 数据标签（用于区分不同类型的数据）
        """
        pass
    
    @abstractmethod
    def pull(self, tag: str = "default") -> Optional[Dict[str, Any]]:
        """
        拉取数据
        
        Args:
            tag: 数据标签
            
        Returns:
            数据字典，无数据时返回None
        """
        pass
    
    @abstractmethod
    def get_version(self, tag: str = "default") -> int:
        """
        获取数据版本号
        
        Args:
            tag: 数据标签
            
        Returns:
            版本号
        """
        pass
