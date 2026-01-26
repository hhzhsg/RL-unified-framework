"""MLP编码器"""
from typing import List
import torch
import torch.nn as nn
from core.interfaces import EncoderInterface
from core.orchestration import register_policy


@register_policy("mlp_encoder")
class MLPEncoder(EncoderInterface):
    """MLP编码器"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, use_layer_norm: bool = True):
        super().__init__()
        
        self._input_dim = input_dim
        self._hidden_dims = hidden_dims
        self._output_dim = output_dim
        
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    @property
    def input_dim(self) -> int:
        return self._input_dim
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    @property
    def latent_dim(self) -> int:
        return self._hidden_dims[-1] if self._hidden_dims else self._input_dim
    
    @property
    def hidden_dims(self) -> List[int]:
        return self._hidden_dims
