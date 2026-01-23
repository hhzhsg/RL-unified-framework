"""
Base Intervention Wrapper

所有干预设备 Wrapper 的抽象基类
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym


class BaseInterventionWrapper(gym.ActionWrapper, ABC):
    """
    干预 Wrapper 抽象基类
    
    职责：
    1. 实时检测人类干预设备输入
    2. 决定使用策略动作还是干预动作
    3. 通过 info 返回干预信息
    
    子类需要实现：
    - get_intervention_action(): 从设备读取干预动作
    - is_intervention_active(): 判断干预是否激活
    
    使用示例：
        env = VRWrapper(base_env)
        action = policy.act(obs)
        obs, reward, done, truncated, info = env.step(action)
        
        if info.get("is_intervention"):
            actual_action = info["intervene_action"]
    """
    
    def __init__(
        self, 
        env: gym.Env,
        intervention_threshold: float = 0.001,
        sticky_duration: float = 0.5,  # 干预后持续时间（秒）
    ):
        """
        Args:
            env: 被包装的环境
            intervention_threshold: 干预激活阈值
            sticky_duration: 干预后的"粘滞"时间，防止抖动
        """
        super().__init__(env)
        self.intervention_threshold = intervention_threshold
        self.sticky_duration = sticky_duration
        
        self._last_intervention_time: float = 0.0
        self._intervention_count: int = 0
        self._episode_intervention_steps: int = 0
    
    @abstractmethod
    def get_intervention_action(self) -> Optional[np.ndarray]:
        """
        从干预设备读取动作
        
        Returns:
            干预动作（如果设备有输入），否则 None
        """
        raise NotImplementedError
    
    @abstractmethod
    def is_intervention_active(self, action: Optional[np.ndarray]) -> bool:
        """
        判断干预是否激活
        
        Args:
            action: 从设备读取的动作
            
        Returns:
            是否激活干预
        """
        raise NotImplementedError
    
    def action(self, policy_action: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        决定实际执行的动作
        
        Args:
            policy_action: 策略输出的动作
            
        Returns:
            (actual_action, is_intervention)
        """
        import time
        
        # 读取干预设备
        intervention_action = self.get_intervention_action()
        
        # 判断干预是否激活
        if intervention_action is not None and self.is_intervention_active(intervention_action):
            self._last_intervention_time = time.time()
            return intervention_action, True
        
        # 粘滞时间内继续使用干预（防止抖动）
        if time.time() - self._last_intervention_time < self.sticky_duration:
            if intervention_action is not None:
                return intervention_action, True
        
        return policy_action, False
    
    def step(self, policy_action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        执行一步环境交互
        
        Args:
            policy_action: 策略输出的动作（总是传入策略动作）
            
        Returns:
            标准 Gym 返回 + 干预信息
        """
        # 决定实际动作
        actual_action, is_intervention = self.action(policy_action)
        
        # 执行动作
        obs, reward, terminated, truncated, info = self.env.step(actual_action)
        
        # 添加干预信息到 info
        info["is_intervention"] = is_intervention
        info["policy_action"] = policy_action
        
        if is_intervention:
            info["intervene_action"] = actual_action
            self._episode_intervention_steps += 1
        
        # 统计信息
        info["episode_intervention_steps"] = self._episode_intervention_steps
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """重置环境"""
        self._episode_intervention_steps = 0
        return self.env.reset(**kwargs)
    
    @property
    def intervention_rate(self) -> float:
        """当前 episode 的干预率"""
        total_steps = getattr(self, '_episode_steps', 1)
        return self._episode_intervention_steps / max(total_steps, 1)
