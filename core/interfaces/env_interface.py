"""
环境接口定义

所有环境实现必须遵循此接口
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple


class EnvInterface(ABC):
    """环境接口"""
    
    @property
    @abstractmethod
    def observation_space(self) -> Dict[str, Any]:
        """观测空间定义"""
        pass
    
    @property
    @abstractmethod
    def action_space(self) -> Dict[str, Any]:
        """动作空间定义"""
        pass
    
    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        重置环境
        
        Returns:
            observation: 初始观测
            info: 额外信息
        """
        pass
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        执行动作
        
        Args:
            action: 动作
            
        Returns:
            observation: 新观测
            reward: 奖励
            terminated: 是否终止
            truncated: 是否截断
            info: 额外信息
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """关闭环境"""
        pass
    
    @property
    def unwrapped(self) -> "EnvInterface":
        """返回未包装的环境"""
        return self
