"""
MLP 策略

包含:
- MLPPolicy: 确定性策略
- MLPGaussianPolicy: 随机策略 (用于 SAC)
"""
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .base import BasePolicy
from data import Observation, RobotState, Action


class MLPPolicy(BasePolicy):
    """
    MLP 确定性策略
    
    输入: robot_state
    输出: action (确定性)
    """
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 hidden_dims: List[int] = [256, 256],
                 activation: str = "relu",
                 action_space: str = "joint"):
        super().__init__(state_dim, action_dim, action_space)
        
        # 构建网络
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            input_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        self.action_head = nn.Linear(input_dim, action_dim)
        
        self._init_weights()
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        return activations.get(name, nn.ReLU())
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # 输出层用更小的增益
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)
    
    def forward(self, obs: Dict[str, torch.Tensor], robot_state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        features = self.backbone(robot_state)
        action = self.action_head(features)
        action = torch.tanh(action)  # 限制到 [-1, 1]
        return action
    
    def act(self, obs: Observation, robot_state: RobotState, 
            deterministic: bool = True) -> Action:
        """推理"""
        self.eval()
        with torch.no_grad():
            state_tensor = torch.from_numpy(robot_state.to_array()).float().unsqueeze(0)
            action_tensor = self.forward({}, state_tensor)
            action_data = action_tensor.squeeze(0).numpy()
        
        return Action(data=action_data, space=self.action_space)


class MLPGaussianPolicy(BasePolicy):
    """
    MLP 随机策略 (用于 SAC)
    
    输出高斯分布，支持重参数化采样
    """
    
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 hidden_dims: List[int] = [256, 256],
                 activation: str = "relu",
                 action_space: str = "joint"):
        super().__init__(state_dim, action_dim, action_space)
        
        # 构建网络
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            input_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(input_dim, action_dim)
        self.log_std_head = nn.Linear(input_dim, action_dim)
        
        self._init_weights()
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        return activations.get(name, nn.ReLU())
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
    
    def forward(self, obs: Dict[str, torch.Tensor], robot_state: torch.Tensor) -> torch.Tensor:
        """前向传播 (返回均值动作)"""
        features = self.backbone(robot_state)
        mean = self.mean_head(features)
        return torch.tanh(mean)
    
    def get_distribution(self, robot_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取动作分布参数"""
        features = self.backbone(robot_state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std
    
    def sample(self, robot_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样动作 (重参数化)
        
        Returns:
            action: 采样的动作
            log_prob: 动作的对数概率
        """
        mean, log_std = self.get_distribution(robot_state)
        std = log_std.exp()
        
        # 重参数化采样
        normal = Normal(mean, std)
        x = normal.rsample()
        
        # Squash 到 [-1, 1]
        action = torch.tanh(x)
        
        # 计算 log_prob (考虑 tanh 变换)
        log_prob = normal.log_prob(x)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def act(self, obs: Observation, robot_state: RobotState, 
            deterministic: bool = True) -> Action:
        """推理"""
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            state_tensor = torch.from_numpy(robot_state.to_array()).float().unsqueeze(0).to(device)
            
            if deterministic:
                action_tensor = self.forward({}, state_tensor)
            else:
                action_tensor, _ = self.sample(state_tensor)
            
            action_data = action_tensor.squeeze(0).cpu().numpy()
        
        return Action(data=action_data, space=self.action_space)
