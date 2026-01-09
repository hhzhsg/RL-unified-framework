"""
VLA-RL Policy 基类
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import torch.nn as nn

from data import Observation, RobotState, Action, Batch


class BasePolicy(ABC, nn.Module):
    """
    策略基类
    
    所有策略实现需要继承此类并实现:
    - forward(): 网络前向传播 (用于训练)
    - act(): 推理输出动作 (用于推理)
    """
    
    def __init__(self, state_dim: int, action_dim: int, action_space: str = "joint"):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = action_space
    
    @abstractmethod
    def forward(self, obs: Dict[str, torch.Tensor], robot_state: torch.Tensor) -> torch.Tensor:
        """
        网络前向传播
        
        Args:
            obs: 观测字典 (batched)，MLP 策略可忽略
            robot_state: 机器人状态 (B, state_dim)
            
        Returns:
            动作输出 (B, action_dim) 或 (B, horizon, action_dim)
        """
        pass
    
    @abstractmethod
    def act(self, obs: Observation, robot_state: RobotState, 
            deterministic: bool = True) -> Action:
        """
        推理输出动作
        
        Args:
            obs: 单个观测
            robot_state: 单个机器人状态
            deterministic: 是否确定性输出
            
        Returns:
            Action 对象
        """
        pass
    
    def compute_loss(self, batch: Batch) -> torch.Tensor:
        """
        计算训练损失 (默认实现为 MSE)
        子类可以覆盖
        """
        pred_action = self.forward(batch.obs, batch.robot_state)
        target_action = batch.action
        
        return torch.nn.functional.mse_loss(pred_action, target_action)
    
    def save(self, path: str):
        """保存模型"""
        torch.save(self.state_dict(), path)
    
    def load(self, path: str, device: str = "cpu"):
        """加载模型"""
        self.load_state_dict(torch.load(path, map_location=device))
