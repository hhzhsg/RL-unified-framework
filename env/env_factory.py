"""
VLA-RL 环境模块 (OpenTau 风格)

借鉴 OpenTau 的设计:
1. configs.py: 环境配置 (dataclass)
2. factory.py: 环境工厂 (注册表模式)
3. 各环境实现: libero.py, dummy.py, ...

设计理念:
- 配置与实现分离
- 注册表模式支持扩展
- 统一的 EnvOutput 接口
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Type
import numpy as np

from data import Action, EnvOutput, Observation, RobotState


# ==================== 配置 (configs.py 风格) ====================

@dataclass
class BaseEnvConfig:
    """环境配置基类"""
    name: str = "base"
    state_dim: int = 16
    action_dim: int = 4
    max_episode_steps: int = 200
    
    # 观测配置
    obs_cameras: List[str] = field(default_factory=lambda: ["front"])
    image_size: int = 224
    
    # 动作空间
    action_space: str = "joint"  # "joint" | "cartesian" | "delta"
    action_low: float = -1.0
    action_high: float = 1.0


@dataclass
class DummyEnvConfig(BaseEnvConfig):
    """DummyEnv 配置"""
    name: str = "dummy"
    deterministic: bool = False
    success_threshold: float = 0.5


@dataclass 
class SimEnvConfig(BaseEnvConfig):
    """仿真环境配置"""
    name: str = "sim"
    physics_dt: float = 0.002
    control_dt: float = 0.05
    render: bool = False
    headless: bool = True


@dataclass
class LiberoEnvConfig(BaseEnvConfig):
    """LIBERO 环境配置"""
    name: str = "libero"
    task_suite: str = "libero_spatial"  # "libero_10" | "libero_goal" | "libero_object" | "libero_spatial"
    task_id: int = 0
    use_camera_obs: bool = True
    camera_names: List[str] = field(default_factory=lambda: ["agentview", "eye_in_hand"])


# ==================== 基类 (base_env.py) ====================

class BaseEnv(ABC):
    """
    环境基类
    
    所有环境需要实现:
    - reset(): 重置环境
    - step(): 执行动作
    """
    
    # 类属性: 环境名称 (用于注册)
    ENV_NAME: str = "base"
    
    def __init__(self, config: BaseEnvConfig):
        self.config = config
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self._step_count = 0
    
    @abstractmethod
    def reset(self, task_id: Optional[str] = None) -> EnvOutput:
        """重置环境"""
        pass
    
    @abstractmethod
    def step(self, action: Action) -> EnvOutput:
        """执行动作"""
        pass
    
    def close(self):
        """关闭环境"""
        pass
    
    def seed(self, seed: int):
        """设置随机种子"""
        np.random.seed(seed)
    
    @property
    def observation_space(self) -> Dict[str, Any]:
        """观测空间"""
        return {
            "cameras": self.config.obs_cameras,
            "image_size": self.config.image_size,
            "state_dim": self.state_dim,
        }
    
    @property
    def action_space(self) -> Dict[str, Any]:
        """动作空间"""
        return {
            "dim": self.action_dim,
            "space": self.config.action_space,
            "low": self.config.action_low,
            "high": self.config.action_high,
        }


# ==================== 工厂 (factory.py 风格) ====================

class EnvRegistry:
    """
    环境注册表
    
    单例模式，管理所有环境类型
    
    Example:
        # 注册环境
        @EnvRegistry.register("dummy")
        class DummyEnv(BaseEnv):
            ...
        
        # 创建环境
        env = EnvRegistry.create("dummy", config)
    """
    
    _instance = None
    _registry: Dict[str, Type[BaseEnv]] = {}
    _config_registry: Dict[str, Type[BaseEnvConfig]] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register(cls, name: str, config_class: Optional[Type[BaseEnvConfig]] = None):
        """
        注册装饰器
        
        Args:
            name: 环境名称
            config_class: 对应的配置类
        """
        def decorator(env_class: Type[BaseEnv]):
            cls._registry[name] = env_class
            if config_class:
                cls._config_registry[name] = config_class
            return env_class
        return decorator
    
    @classmethod
    def register_env(cls, name: str, env_class: Type[BaseEnv], 
                     config_class: Optional[Type[BaseEnvConfig]] = None):
        """直接注册环境"""
        cls._registry[name] = env_class
        if config_class:
            cls._config_registry[name] = config_class
    
    @classmethod
    def create(cls, name: str, config: Optional[BaseEnvConfig] = None, **kwargs) -> BaseEnv:
        """
        创建环境实例
        
        Args:
            name: 环境名称
            config: 环境配置
            **kwargs: 传递给环境构造函数的额外参数
            
        Returns:
            环境实例
        """
        if name not in cls._registry:
            raise ValueError(
                f"未知环境: {name}。"
                f"可用环境: {list(cls._registry.keys())}"
            )
        
        env_class = cls._registry[name]
        
        # 如果没有提供配置，使用默认配置
        if config is None:
            if name in cls._config_registry:
                config = cls._config_registry[name]()
            else:
                config = BaseEnvConfig(name=name)
        
        return env_class(config, **kwargs)
    
    @classmethod
    def list_envs(cls) -> List[str]:
        """列出所有注册的环境"""
        return list(cls._registry.keys())
    
    @classmethod
    def get_config_class(cls, name: str) -> Optional[Type[BaseEnvConfig]]:
        """获取环境对应的配置类"""
        return cls._config_registry.get(name)


# 全局实例
ENV_REGISTRY = EnvRegistry()


# ==================== 便捷函数 ====================

def create_env(config: BaseEnvConfig, **kwargs) -> BaseEnv:
    """
    创建环境
    
    Args:
        config: 环境配置
        **kwargs: 额外参数
        
    Returns:
        环境实例
    """
    return ENV_REGISTRY.create(config.name, config, **kwargs)


def register_env(name: str, config_class: Optional[Type[BaseEnvConfig]] = None):
    """
    环境注册装饰器
    
    Example:
        @register_env("my_env", MyEnvConfig)
        class MyEnv(BaseEnv):
            ...
    """
    return ENV_REGISTRY.register(name, config_class)


# ==================== 内置环境实现 ====================

@register_env("dummy", DummyEnvConfig)
class DummyEnv(BaseEnv):
    """
    虚拟环境 - 用于测试
    
    简单的目标到达任务
    """
    
    ENV_NAME = "dummy"
    
    def __init__(self, config: DummyEnvConfig, 
                 reward_fn: Optional[Callable] = None):
        super().__init__(config)
        
        self.deterministic = config.deterministic
        self.success_threshold = config.success_threshold
        self.reward_fn = reward_fn
        
        self._current_state = np.zeros(self.state_dim, dtype=np.float32)
        self._target_state = np.zeros(self.state_dim, dtype=np.float32)
    
    def reset(self, task_id: Optional[str] = None) -> EnvOutput:
        self._step_count = 0
        
        if self.deterministic:
            self._current_state = np.zeros(self.state_dim, dtype=np.float32)
            self._target_state = np.ones(self.state_dim, dtype=np.float32)
        else:
            self._current_state = np.random.randn(self.state_dim).astype(np.float32) * 0.1
            self._target_state = np.random.randn(self.state_dim).astype(np.float32)
        
        return EnvOutput(
            obs=self._get_observation(),
            robot_state=RobotState(raw_state=self._current_state.copy()),
            reward=0.0,
            done=False,
            info={"task_id": task_id or "reach_target"},
        )
    
    def step(self, action: Action) -> EnvOutput:
        self._step_count += 1
        
        action_data = np.asarray(action.data).flatten()[:self.action_dim]
        
        # 状态转移
        scale = 0.1
        noise = 0 if self.deterministic else np.random.randn(self.state_dim).astype(np.float32) * 0.01
        
        affected_dims = min(len(action_data), self.state_dim)
        state_delta = np.zeros(self.state_dim, dtype=np.float32)
        state_delta[:affected_dims] = action_data[:affected_dims] * scale
        
        self._current_state = self._current_state + state_delta + noise
        
        # 奖励
        if self.reward_fn:
            reward = self.reward_fn(self._current_state, action_data, self._target_state)
        else:
            dist = np.linalg.norm(self._current_state - self._target_state)
            reward = -dist * 0.1
        
        # 结束条件
        dist_to_target = np.linalg.norm(self._current_state - self._target_state)
        success = dist_to_target < self.success_threshold
        done = self._step_count >= self.config.max_episode_steps or success
        
        if success:
            reward = 10.0
        
        return EnvOutput(
            obs=self._get_observation(),
            robot_state=RobotState(raw_state=self._current_state.copy()),
            reward=float(reward),
            done=done,
            info={"success": success, "step": self._step_count, "dist": dist_to_target},
        )
    
    def _get_observation(self) -> Observation:
        images = {}
        if not self.deterministic:
            for cam in self.config.obs_cameras:
                images[cam] = np.random.randint(
                    0, 256, (self.config.image_size, self.config.image_size, 3), dtype=np.uint8
                )
        return Observation(images=images, language="reach the target")


# ==================== 向量化环境 ====================

class VectorEnv:
    """
    向量化环境 - 并行运行多个环境
    
    用于加速数据采集
    """
    
    def __init__(self, env_fns: List[Callable[[], BaseEnv]]):
        """
        Args:
            env_fns: 环境创建函数列表
        """
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
    
    def reset(self) -> List[EnvOutput]:
        """重置所有环境"""
        return [env.reset() for env in self.envs]
    
    def step(self, actions: List[Action]) -> List[EnvOutput]:
        """在所有环境执行动作"""
        return [env.step(action) for env, action in zip(self.envs, actions)]
    
    def close(self):
        """关闭所有环境"""
        for env in self.envs:
            env.close()


def make_vec_env(name: str, num_envs: int, config: Optional[BaseEnvConfig] = None) -> VectorEnv:
    """
    创建向量化环境
    
    Args:
        name: 环境名称
        num_envs: 环境数量
        config: 环境配置
        
    Returns:
        VectorEnv 实例
    """
    env_fns = [lambda: ENV_REGISTRY.create(name, config) for _ in range(num_envs)]
    return VectorEnv(env_fns)


# ==================== 导出 ====================

__all__ = [
    # 配置
    "BaseEnvConfig",
    "DummyEnvConfig",
    "SimEnvConfig",
    "LiberoEnvConfig",
    # 基类
    "BaseEnv",
    # 工厂
    "EnvRegistry",
    "ENV_REGISTRY",
    "create_env",
    "register_env",
    # 实现
    "DummyEnv",
    # 向量化
    "VectorEnv",
    "make_vec_env",
]
