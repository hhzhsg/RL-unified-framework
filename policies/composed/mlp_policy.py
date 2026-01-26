"""
简单 MLP 策略

用于 HIL 训练流程验证，不需要完整的 SAC 结构
仅包含一个 MLP Actor，输出确定性动作
"""
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import numpy as np

from policies.base_policy import BasePolicy
from core.orchestration import register_policy


@register_policy("mlp")
class MLPPolicy(BasePolicy):
    """
    简单 MLP 策略
    
    用于 HIL 训练流程验证
    输入：state
    输出：确定性动作
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        hidden_dims = config.get("hidden_dims", [256, 256])
        activation = config.get("activation", "relu")
        
        # 构建 MLP
        layers = []
        in_dim = self.state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, self.action_dim))
        layers.append(nn.Tanh())  # 输出范围 [-1, 1]
        
        self.net = nn.Sequential(*layers)
        
        # Action scaling
        self.action_scale = config.get("action_scale", 1.0)
    
    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播"""
        if isinstance(obs, dict):
            state = obs.get("state", obs.get("observation", obs.get("qpos")))
        else:
            state = obs
        
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        state = state.to(self.device)
        action = self.net(state)
        return action * self.action_scale
    
    def act(self, obs: Dict[str, Any], deterministic: bool = False) -> np.ndarray:
        """推理动作"""
        with torch.no_grad():
            action = self.forward(obs)
            action = action.squeeze(0).cpu().numpy()
        return action
    
    def reset(self) -> None:
        """重置状态（MLP 无状态）"""
        pass
