"""
价值网络

包含:
- QNetwork: Q 函数网络 Q(s, a)
- VNetwork: V 函数网络 V(s)
"""
from typing import List
import numpy as np
import torch
import torch.nn as nn

from .base import BaseNetwork


class QNetwork(BaseNetwork):
    """
    Q 函数网络
    
    Q(s, a) -> scalar
    """
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 hidden_dims: List[int] = [256, 256],
                 activation: str = "relu"):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 构建网络
        layers = []
        input_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            input_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        self.q_head = nn.Linear(input_dim, 1)
        
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
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 状态 (batch_size, state_dim)
            action: 动作 (batch_size, action_dim)
            
        Returns:
            Q 值 (batch_size, 1)
        """
        x = torch.cat([state, action], dim=-1)
        features = self.backbone(x)
        q_value = self.q_head(features)
        return q_value


class VNetwork(BaseNetwork):
    """
    V 函数网络
    
    V(s) -> scalar
    """
    
    def __init__(self, 
                 state_dim: int, 
                 hidden_dims: List[int] = [256, 256],
                 activation: str = "relu"):
        super().__init__()
        
        self.state_dim = state_dim
        
        # 构建网络
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            input_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        self.v_head = nn.Linear(input_dim, 1)
        
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
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 状态 (batch_size, state_dim)
            
        Returns:
            V 值 (batch_size, 1)
        """
        features = self.backbone(state)
        v_value = self.v_head(features)
        return v_value
