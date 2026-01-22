"""
Policy接口定义

策略网络必须遵循此接口
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn


class PolicyInterface(nn.Module, ABC):
    """策略接口"""
    
    @abstractmethod
    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            obs: 观测字典
            
        Returns:
            动作或动作分布参数
        """
        pass
    
    @abstractmethod
    def act(self, obs: Dict[str, Any], deterministic: bool = False) -> Any:
        """
        推理动作（单步）
        
        Args:
            obs: 观测
            deterministic: 是否确定性动作
            
        Returns:
            动作
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """重置策略状态（用于RNN等有状态策略）"""
        pass
    
    @property
    @abstractmethod
    def device(self) -> torch.device:
        """策略所在设备"""
        pass
    
    def to_device(self, device: torch.device) -> "PolicyInterface":
        """移动到指定设备"""
        return self.to(device)


class ActorInterface(PolicyInterface):
    """Actor策略接口"""
    
    @abstractmethod
    def sample(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样动作（带log_prob）
        
        Args:
            obs: 观测
            
        Returns:
            action: 动作
            log_prob: 动作对数概率
        """
        pass
    
    @abstractmethod
    def log_prob(self, obs: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        """计算动作的对数概率"""
        pass


class CriticInterface(PolicyInterface):
    """Critic策略接口"""
    
    @abstractmethod
    def forward(self, obs: Dict[str, torch.Tensor], action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算Q值或V值
        
        Args:
            obs: 观测
            action: 动作（Q-Critic需要，V-Critic不需要）
            
        Returns:
            value: Q值或V值
        """
        pass
