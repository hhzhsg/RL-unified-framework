"""
统一注册表

集中管理所有可注册组件
"""
from typing import Dict, Type, Any, Optional


class Registry:
    """
    统一注册表
    
    支持注册和获取:
    - 策略 (policy)
    - 算法 (algorithm)
    - 缓冲区 (buffer)
    - 环境 (env)
    - 机器人适配器 (robot)
    
    Example:
        @REGISTRY.register_policy("my_policy")
        class MyPolicy(BasePolicy):
            ...
        
        policy_cls = REGISTRY.get_policy("my_policy")
    """
    
    def __init__(self):
        self._policies: Dict[str, Type] = {}
        self._algorithms: Dict[str, Type] = {}
        self._buffers: Dict[str, Type] = {}
        self._envs: Dict[str, Type] = {}
        self._robots: Dict[str, Type] = {}
    
    # ==================== Policy ====================
    
    def register_policy(self, name: str):
        """注册策略的装饰器"""
        def decorator(cls):
            self._policies[name] = cls
            return cls
        return decorator
    
    def get_policy(self, name: str) -> Type:
        """获取策略类"""
        if name not in self._policies:
            raise KeyError(f"Policy '{name}' not found. Available: {list(self._policies.keys())}")
        return self._policies[name]
    
    @property
    def policies(self) -> Dict[str, Type]:
        return self._policies.copy()
    
    # ==================== Algorithm ====================
    
    def register_algorithm(self, name: str):
        """注册算法的装饰器"""
        def decorator(cls):
            self._algorithms[name] = cls
            return cls
        return decorator
    
    def get_algorithm(self, name: str) -> Type:
        """获取算法类"""
        if name not in self._algorithms:
            raise KeyError(f"Algorithm '{name}' not found. Available: {list(self._algorithms.keys())}")
        return self._algorithms[name]
    
    @property
    def algorithms(self) -> Dict[str, Type]:
        return self._algorithms.copy()
    
    # ==================== Buffer ====================
    
    def register_buffer(self, name: str):
        """注册缓冲区的装饰器"""
        def decorator(cls):
            self._buffers[name] = cls
            return cls
        return decorator
    
    def get_buffer(self, name: str) -> Type:
        """获取缓冲区类"""
        if name not in self._buffers:
            raise KeyError(f"Buffer '{name}' not found. Available: {list(self._buffers.keys())}")
        return self._buffers[name]
    
    @property
    def buffers(self) -> Dict[str, Type]:
        return self._buffers.copy()
    
    # ==================== Env ====================
    
    def register_env(self, name: str):
        """注册环境的装饰器"""
        def decorator(cls):
            self._envs[name] = cls
            return cls
        return decorator
    
    def get_env(self, name: str) -> Type:
        """获取环境类"""
        if name not in self._envs:
            raise KeyError(f"Env '{name}' not found. Available: {list(self._envs.keys())}")
        return self._envs[name]
    
    @property
    def envs(self) -> Dict[str, Type]:
        return self._envs.copy()
    
    # ==================== Robot ====================
    
    def register_robot(self, name: str):
        """注册机器人适配器的装饰器"""
        def decorator(cls):
            self._robots[name] = cls
            return cls
        return decorator
    
    def get_robot(self, name: str) -> Type:
        """获取机器人适配器类"""
        if name not in self._robots:
            raise KeyError(f"Robot '{name}' not found. Available: {list(self._robots.keys())}")
        return self._robots[name]
    
    @property
    def robots(self) -> Dict[str, Type]:
        return self._robots.copy()
    
    # ==================== 工厂方法 ====================
    
    def create(self, category: str, name: str, *args, **kwargs) -> Any:
        """
        通用创建方法
        
        Args:
            category: 类别 ("policy", "algorithm", "buffer", "env", "robot")
            name: 组件名称
            *args, **kwargs: 传递给构造函数的参数
            
        Returns:
            组件实例
        """
        getters = {
            "policy": self.get_policy,
            "algorithm": self.get_algorithm,
            "buffer": self.get_buffer,
            "env": self.get_env,
            "robot": self.get_robot,
        }
        
        if category not in getters:
            raise ValueError(f"Unknown category: {category}")
        
        cls = getters[category](name)
        return cls(*args, **kwargs)


# 全局注册表实例
REGISTRY = Registry()


def _auto_register():
    """自动注册已有组件"""
    # 策略
    from policy import MLPPolicy, MLPGaussianPolicy
    REGISTRY._policies.update({
        "mlp": MLPPolicy,
        "mlp_gaussian": MLPGaussianPolicy,
    })
    
    # 算法
    from algorithm import BC, SAC, TD3BC
    REGISTRY._algorithms.update({
        "bc": BC,
        "sac": SAC,
        "td3bc": TD3BC,
    })
    
    # 缓冲区
    from buffer import ReplayBuffer
    REGISTRY._buffers.update({
        "replay": ReplayBuffer,
    })
    
    # 环境
    from env import DummyEnv
    REGISTRY._envs.update({
        "dummy": DummyEnv,
    })
    
    # 机器人
    from robot import BinocularAdapter
    REGISTRY._robots.update({
        "binocular": BinocularAdapter,
    })


# 延迟注册，避免循环导入
def get_registry() -> Registry:
    """获取已初始化的注册表"""
    if not REGISTRY._policies:
        _auto_register()
    return REGISTRY
