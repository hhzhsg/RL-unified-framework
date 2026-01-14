"""
数据转换基类

提供统一的数据预处理接口，支持链式组合
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable
import numpy as np


class BaseTransform(ABC):
    """转换基类"""
    
    @abstractmethod
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行转换
        
        Args:
            data: 输入数据字典
            
        Returns:
            转换后的数据字典
        """
        pass
    
    @property
    def name(self) -> str:
        return self.__class__.__name__


class Compose(BaseTransform):
    """
    组合多个转换
    
    Example:
        transform = Compose([
            ResizeImage(224),
            NormalizeState(),
            ToTensor(),
        ])
        output = transform(data)
    """
    
    def __init__(self, transforms: List[BaseTransform]):
        self.transforms = transforms
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for t in self.transforms:
            data = t(data)
        return data
    
    def __repr__(self) -> str:
        names = [t.name for t in self.transforms]
        return f"Compose({names})"


class Identity(BaseTransform):
    """恒等变换 (不做任何处理)"""
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return data


class Lambda(BaseTransform):
    """自定义函数变换"""
    
    def __init__(self, fn: Callable[[Dict[str, Any]], Dict[str, Any]], name: str = "Lambda"):
        self.fn = fn
        self._name = name
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self.fn(data)
    
    @property
    def name(self) -> str:
        return self._name
