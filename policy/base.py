"""
Policy 基类

所有策略网络的抽象基类
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn

from data import Observation, RobotState, Action


class BasePolicy(nn.Module, ABC):
    """
    策略基类
    
    所有策略需要实现:
    - forward(): 前向传播，返回动作 Tensor
    - act(): 推理，返回 Action 对象
    """
    
    def __init__(self, state_dim: int, action_dim: int, action_space: str = "joint"):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            action_space: 动作空间类型 ("joint" | "cartesian" | "delta")
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = action_space
    
    @abstractmethod
    def forward(self, obs: Dict[str, torch.Tensor], robot_state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            obs: 观测字典 (图像等)
            robot_state: 机器人状态 (batch_size, state_dim)
            
        Returns:
            动作 Tensor (batch_size, action_dim)
        """
        pass
    
    @abstractmethod
    def act(self, obs: Observation, robot_state: RobotState, 
            deterministic: bool = True) -> Action:
        """
        推理 (单步)
        
        Args:
            obs: 观测
            robot_state: 机器人状态
            deterministic: 是否确定性动作
            
        Returns:
            动作
        """
        pass
    
    def reset(self):
        """重置状态 (用于 RNN 等有状态策略)"""
        pass
    
    def get_action_dim(self) -> int:
        return self.action_dim
    
    def get_state_dim(self) -> int:
        return self.state_dim
