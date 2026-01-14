"""
环境基类

所有环境的抽象基类
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from data import Action, EnvOutput
from config import EnvConfig


class BaseEnv(ABC):
    """
    环境基类
    
    所有环境需要实现:
    - reset(): 重置环境
    - step(): 执行动作
    """
    
    def __init__(self, config: EnvConfig):
        """
        Args:
            config: 环境配置
        """
        self.config = config
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self._step_count = 0
    
    @abstractmethod
    def reset(self, task_id: Optional[str] = None) -> EnvOutput:
        """
        重置环境
        
        Args:
            task_id: 任务 ID (用于多任务环境)
            
        Returns:
            初始环境输出
        """
        pass
    
    @abstractmethod
    def step(self, action: Action) -> EnvOutput:
        """
        执行动作
        
        Args:
            action: 动作
            
        Returns:
            环境输出
        """
        pass
    
    def close(self):
        """关闭环境"""
        pass
    
    def seed(self, seed: int):
        """设置随机种子"""
        pass
    
    @property
    def max_episode_steps(self) -> int:
        return self.config.max_episode_steps
    
    @property
    def step_count(self) -> int:
        return self._step_count
