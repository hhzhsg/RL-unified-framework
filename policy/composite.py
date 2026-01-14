"""
组合策略

支持在预训练模型基础上进行微调:
- ResidualPolicy: 残差策略
- EnsemblePolicy: 集成策略
"""
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn

from .base import BasePolicy
from data import Observation, RobotState, Action


class ResidualPolicy(BasePolicy):
    """
    残差策略
    
    action = base_policy(obs) + residual_scale * residual_policy(obs)
    
    用于在预训练 VLA 基础上学习残差修正
    """
    
    def __init__(self, 
                 base_policy: BasePolicy, 
                 residual_policy: BasePolicy,
                 residual_scale: float = 0.1):
        """
        Args:
            base_policy: 基础策略 (通常冻结)
            residual_policy: 残差策略 (训练)
            residual_scale: 残差缩放系数
        """
        super().__init__(
            state_dim=base_policy.state_dim,
            action_dim=base_policy.action_dim,
            action_space=base_policy.action_space
        )
        
        self.base_policy = base_policy
        self.residual_policy = residual_policy
        self.residual_scale = residual_scale
    
    def forward(self, obs: Dict[str, torch.Tensor], robot_state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # Base action (不计算梯度)
        with torch.no_grad():
            base_action = self.base_policy.forward(obs, robot_state)
        
        # Residual
        residual = self.residual_policy.forward(obs, robot_state)
        
        # 组合
        action = base_action + self.residual_scale * residual
        action = torch.clamp(action, -1, 1)
        
        return action
    
    def act(self, obs: Observation, robot_state: RobotState,
            deterministic: bool = True) -> Action:
        """推理"""
        self.eval()
        with torch.no_grad():
            state_tensor = torch.from_numpy(robot_state.to_array()).float().unsqueeze(0)
            obs_dict = {}  # 简化版本
            
            action_tensor = self.forward(obs_dict, state_tensor)
            action_data = action_tensor.squeeze(0).numpy()
        
        return Action(data=action_data, space=self.action_space)
    
    def freeze_base(self):
        """冻结基础策略"""
        for param in self.base_policy.parameters():
            param.requires_grad = False
    
    def unfreeze_base(self):
        """解冻基础策略"""
        for param in self.base_policy.parameters():
            param.requires_grad = True


class EnsemblePolicy(BasePolicy):
    """
    集成策略
    
    对多个策略的输出取加权平均
    """
    
    def __init__(self, policies: List[BasePolicy], weights: List[float] = None):
        """
        Args:
            policies: 策略列表
            weights: 权重列表，None 表示均匀权重
        """
        if len(policies) == 0:
            raise ValueError("At least one policy required")
        
        super().__init__(
            state_dim=policies[0].state_dim,
            action_dim=policies[0].action_dim,
            action_space=policies[0].action_space
        )
        
        self.policies = nn.ModuleList(policies)
        
        if weights is None:
            weights = [1.0 / len(policies)] * len(policies)
        self.weights = weights
    
    def forward(self, obs: Dict[str, torch.Tensor], robot_state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        actions = []
        for policy, weight in zip(self.policies, self.weights):
            action = policy.forward(obs, robot_state)
            actions.append(action * weight)
        
        return sum(actions)
    
    def act(self, obs: Observation, robot_state: RobotState,
            deterministic: bool = True) -> Action:
        """推理"""
        self.eval()
        with torch.no_grad():
            state_tensor = torch.from_numpy(robot_state.to_array()).float().unsqueeze(0)
            action_tensor = self.forward({}, state_tensor)
            action_data = action_tensor.squeeze(0).numpy()
        
        return Action(data=action_data, space=self.action_space)
