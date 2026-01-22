"""
Buffer接口定义

所有数据缓冲区实现必须遵循此接口
"""
from abc import ABC, abstractmethod
from typing import List, Any, Optional, Dict


class BufferInterface(ABC):
    """Buffer接口"""
    
    @abstractmethod
    def add(self, data: Dict[str, Any]) -> None:
        """
        添加单条数据
        
        Args:
            data: 数据字典，包含 obs, action, reward, next_obs, done 等
        """
        pass
    
    @abstractmethod
    def add_batch(self, data: Dict[str, Any]) -> None:
        """
        批量添加数据
        
        Args:
            data: 批量数据字典
        """
        pass
    
    @abstractmethod
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """
        采样数据
        
        Args:
            batch_size: 批量大小
            
        Returns:
            采样的批量数据
        """
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """返回buffer中的数据量"""
        pass
    
    @property
    @abstractmethod
    def capacity(self) -> int:
        """返回buffer容量"""
        pass
    
    def clear(self) -> None:
        """清空buffer"""
        pass
    
    def save(self, path: str) -> None:
        """保存buffer到文件"""
        pass
    
    def load(self, path: str) -> None:
        """从文件加载buffer"""
        pass
