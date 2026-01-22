"""
组件注册表

管理所有可用组件的注册和获取
"""
from typing import Dict, Type, Any, Optional, Callable


class ComponentRegistry:
    """
    组件注册表
    
    集中管理所有可注册组件:
    - env: 环境
    - buffer: 缓冲区
    - sampler: 采样器
    - policy: 策略
    - algorithm: 算法
    - sync: 同步器
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registries = {
                "env": {},
                "buffer": {},
                "sampler": {},
                "policy": {},
                "algorithm": {},
                "sync": {},
                "transform": {},
            }
        return cls._instance
    
    def register(self, category: str, name: str, cls_or_factory: Type | Callable) -> None:
        """
        注册组件
        
        Args:
            category: 组件类别
            name: 组件名称
            cls_or_factory: 组件类或工厂函数
        """
        if category not in self._registries:
            raise ValueError(f"Unknown category: {category}")
        self._registries[category][name] = cls_or_factory
    
    def get(self, category: str, name: str) -> Type | Callable:
        """
        获取组件
        
        Args:
            category: 组件类别
            name: 组件名称
            
        Returns:
            组件类或工厂函数
        """
        if category not in self._registries:
            raise ValueError(f"Unknown category: {category}")
        if name not in self._registries[category]:
            available = list(self._registries[category].keys())
            raise ValueError(f"Unknown {category}: {name}. Available: {available}")
        return self._registries[category][name]
    
    def list(self, category: str) -> list:
        """列出某类别的所有组件"""
        if category not in self._registries:
            raise ValueError(f"Unknown category: {category}")
        return list(self._registries[category].keys())
    
    def list_all(self) -> Dict[str, list]:
        """列出所有组件"""
        return {cat: list(reg.keys()) for cat, reg in self._registries.items()}


# 全局注册表实例
REGISTRY = ComponentRegistry()


def register_env(name: str):
    """环境注册装饰器"""
    def decorator(cls):
        REGISTRY.register("env", name, cls)
        return cls
    return decorator


def register_buffer(name: str):
    """Buffer注册装饰器"""
    def decorator(cls):
        REGISTRY.register("buffer", name, cls)
        return cls
    return decorator


def register_sampler(name: str):
    """Sampler注册装饰器"""
    def decorator(cls):
        REGISTRY.register("sampler", name, cls)
        return cls
    return decorator


def register_policy(name: str):
    """Policy注册装饰器"""
    def decorator(cls):
        REGISTRY.register("policy", name, cls)
        return cls
    return decorator


def register_algorithm(name: str):
    """Algorithm注册装饰器"""
    def decorator(cls):
        REGISTRY.register("algorithm", name, cls)
        return cls
    return decorator


def register_sync(name: str):
    """同步器注册装饰器"""
    def decorator(cls):
        REGISTRY.register("sync", name, cls)
        return cls
    return decorator


def register_transform(name: str):
    """Transform注册装饰器"""
    def decorator(cls):
        REGISTRY.register("transform", name, cls)
        return cls
    return decorator
