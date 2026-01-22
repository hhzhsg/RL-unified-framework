"""
环境基类
"""
from abc import abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np

from core.interfaces import EnvInterface
from core.orchestration import register_env


class BaseEnv(EnvInterface):
    """
    环境基类
    
    所有环境实现需要继承此类
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._step_count = 0
        self._episode_count = 0
    
    @property
    @abstractmethod
    def observation_space(self) -> Dict[str, Any]:
        """观测空间"""
        pass
    
    @property
    @abstractmethod
    def action_space(self) -> Dict[str, Any]:
        """动作空间"""
        pass
    
    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """重置环境"""
        pass
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """执行动作"""
        pass
    
    def close(self) -> None:
        """关闭环境"""
        pass
    
    def seed(self, seed: int) -> None:
        """设置随机种子"""
        pass
    
    @property
    def step_count(self) -> int:
        return self._step_count
    
    @property
    def episode_count(self) -> int:
        return self._episode_count
