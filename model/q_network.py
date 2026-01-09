"""
VLA-RL Q Network
用于 TD3+BC, CQL, IQL 等 Offline RL 算法
"""
from typing import List
import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Q(s, a) 网络
    输入: state + action
    输出: Q 值 (标量)
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        input_dim = state_dim + action_dim
        
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [B, state_dim]
            action: [B, action_dim]
        Returns:
            q_value: [B]
        """
        x = torch.cat([state, action], dim=-1)
        return self.network(x).squeeze(-1)


class VNetwork(nn.Module):
    """
    V(s) 网络 (用于 IQL)
    输入: state
    输出: V 值 (标量)
    """
    
    def __init__(self, state_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        input_dim = state_dim
        
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [B, state_dim]
        Returns:
            v_value: [B]
        """
        return self.network(state).squeeze(-1)
