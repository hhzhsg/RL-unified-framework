"""
VLA-RL 环境基类
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from data import Observation, RobotState, Action, EnvOutput
from config import EnvConfig


class BaseEnv(ABC):
    """
    环境基类
    
    所有环境实现需要继承此类并实现:
    - reset(): 重置环境
    - step(): 执行动作
    """
    
    def __init__(self, config: EnvConfig):
        self.config = config
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self._step_count = 0
    
    @abstractmethod
    def reset(self, task_id: Optional[str] = None) -> EnvOutput:
        """
        重置环境
        
        Args:
            task_id: 可选的任务标识
            
        Returns:
            EnvOutput: 包含 obs, robot_state, reward, done, info
        """
        pass
    
    @abstractmethod
    def step(self, action: Action) -> EnvOutput:
        """
        执行动作
        
        Args:
            action: Action 对象
            
        Returns:
            EnvOutput: 包含 obs, robot_state, reward, done, info
        """
        pass
    
    def close(self):
        """关闭环境，释放资源"""
        pass
    
    @property
    def observation_space(self) -> Dict[str, Any]:
        """观测空间描述"""
        return {
            "cameras": self.config.obs_cameras,
            "image_size": self.config.image_size,
        }
    
    @property
    def action_space(self) -> Dict[str, Any]:
        """动作空间描述"""
        return {
            "dim": self.action_dim,
            "space": self.config.action_space,
            "low": -1.0,
            "high": 1.0,
        }
