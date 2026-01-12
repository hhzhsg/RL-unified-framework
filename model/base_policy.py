"""
VLA-RL Policy 基类

所有策略网络继承此类
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

from data import Observation, RobotState, Action


class BasePolicy(nn.Module, ABC):
    """
    策略基类
    
    所有策略需要实现:
    - forward(): 前向传播，返回动作
    - act(): 推理时使用，返回 Action 对象
    """
    
    def __init__(self, state_dim: int, action_dim: int, action_space: str = "joint"):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = action_space
    
    @abstractmethod
    def forward(self, obs: Dict[str, torch.Tensor], robot_state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            obs: 观测字典 (可包含图像等)
            robot_state: 机器人状态 (B, state_dim)
            
        Returns:
            action: 动作 (B, action_dim)
        """
        pass
    
    @abstractmethod
    def act(self, obs: Observation, robot_state: RobotState, 
            deterministic: bool = True) -> Action:
        """
        推理接口
        
        Args:
            obs: Observation 对象
            robot_state: RobotState 对象
            deterministic: 是否确定性输出
            
        Returns:
            Action 对象
        """
        pass
