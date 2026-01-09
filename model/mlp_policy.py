"""
VLA-RL MLP Policy 实现

两种策略:
- MLPPolicy: 确定性策略，输出动作
- MLPGaussianPolicy: 随机策略，输出动作分布 (用于 SAC)
"""
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn

from .base_policy import BasePolicy
from data import Observation, RobotState, Action


class MLPPolicy(BasePolicy):
    """
    MLP 确定性策略
    
    输入: robot_state
    输出: action (确定性)
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
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
            
            if not deterministic:
                noise = torch.randn_like(action_tensor) * 0.1
                action_tensor = torch.clamp(action_tensor + noise, -1, 1)
            
            action_data = action_tensor.squeeze(0).numpy()
        
        return Action(data=action_data, space=self.action_space)


class MLPGaussianPolicy(BasePolicy):
    """
    高斯策略 (用于 SAC)
    
    输入: robot_state
    输出: 动作的均值和标准差，支持采样
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 activation: str = "relu",
                 action_space: str = "joint",
                 log_std_min: float = -20,
                 log_std_max: float = 2):
        super().__init__(state_dim, action_dim, action_space)
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
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
        }
        return activations.get(name, nn.ReLU())
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
    
    def forward(self, obs: Dict[str, torch.Tensor], robot_state: torch.Tensor):
        """返回均值和 log_std"""
        features = self.backbone(robot_state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, obs: Dict[str, torch.Tensor], robot_state: torch.Tensor):
        """
        采样动作 (带重参数化)
        
        Returns:
            action: 采样的动作 (B, action_dim)
            log_prob: 对数概率 (B, 1)
        """
        mean, log_std = self.forward(obs, robot_state)
        std = log_std.exp()
        
        # 重参数化采样
        dist = torch.distributions.Normal(mean, std)
        action_raw = dist.rsample()
        
        # Squash to [-1, 1]
        action = torch.tanh(action_raw)
        
        # 计算 log_prob (考虑 tanh 变换)
        log_prob = dist.log_prob(action_raw)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def act(self, obs: Observation, robot_state: RobotState,
            deterministic: bool = True) -> Action:
        """推理"""
        self.eval()
        with torch.no_grad():
            state_tensor = torch.from_numpy(robot_state.to_array()).float().unsqueeze(0)
            
            if deterministic:
                mean, _ = self.forward({}, state_tensor)
                action_tensor = torch.tanh(mean)
            else:
                action_tensor, _ = self.sample({}, state_tensor)
            
            action_data = action_tensor.squeeze(0).numpy()
        
        return Action(data=action_data, space=self.action_space)
