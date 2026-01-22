"""
Transform接口定义

数据变换接口
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List


class TransformInterface(ABC):
    """数据变换接口"""
    
    @abstractmethod
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行变换
        
        Args:
            data: 输入数据字典
            
        Returns:
            变换后的数据字典
        """
        pass
    
    @property
    def name(self) -> str:
        """变换名称"""
        return self.__class__.__name__


class ComposableTransform(TransformInterface):
    """可组合变换"""
    
    def __init__(self, transforms: List[TransformInterface]):
        self.transforms = transforms
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for t in self.transforms:
            data = t(data)
        return data
    
    @property
    def name(self) -> str:
        names = [t.name for t in self.transforms]
        return f"Compose({names})"


class ReversibleTransform(TransformInterface):
    """可逆变换接口"""
    
    @abstractmethod
    def inverse(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """逆变换"""
        pass
