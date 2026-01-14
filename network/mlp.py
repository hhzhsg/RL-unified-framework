"""
MLP 网络

通用 MLP 网络构建块
"""
from typing import List, Optional, Callable
import numpy as np
import torch
import torch.nn as nn

from .base import BaseNetwork


class MLP(BaseNetwork):
    """
    通用 MLP 网络
    
    Example:
        mlp = MLP(
            input_dim=64,
            output_dim=10,
            hidden_dims=[256, 256],
            activation="relu",
        )
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 activation: str = "relu",
                 output_activation: Optional[str] = None,
                 dropout: float = 0.0,
                 layer_norm: bool = False):
        """
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            hidden_dims: 隐藏层维度列表
            activation: 激活函数
            output_activation: 输出层激活函数
            dropout: Dropout 比例
            layer_norm: 是否使用 LayerNorm
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 构建网络
        layers = []
        dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self._get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        self.output_layer = nn.Linear(dim, output_dim)
        
        self.output_activation = None
        if output_activation:
            self.output_activation = self._get_activation(output_activation)
        
        self._init_weights()
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "sigmoid": nn.Sigmoid(),
            "softmax": nn.Softmax(dim=-1),
        }
        return activations.get(name, nn.ReLU())
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        features = self.backbone(x)
        output = self.output_layer(features)
        
        if self.output_activation:
            output = self.output_activation(output)
        
        return output
