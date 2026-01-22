"""高斯Actor"""
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import numpy as np
from policies.base_policy import BasePolicy
from core.interfaces import ActorInterface
from core.orchestration import register_policy


@register_policy("gaussian_actor")
class GaussianActor(BasePolicy, ActorInterface):
    """高斯策略Actor"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        hidden_dims = config.get("hidden_dims", [256, 256])
        
        # 构建网络
        layers = []
        prev_dim = self.state_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
            ])
            prev_dim = dim
        
        self.net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev_dim, self.action_dim)
        self.log_std_head = nn.Linear(prev_dim, self.action_dim)
        
        self.log_std_min = config.get("log_std_min", -20)
        self.log_std_max = config.get("log_std_max", 2)
    
    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        state = obs["state"]
        features = self.net(state)
        mean = self.mean_head(features)
        return mean
    
    def sample(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        state = obs["state"]
        features = self.net(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        # 重参数化采样
        noise = torch.randn_like(mean)
        action = mean + std * noise
        
        # Tanh squash
        action = torch.tanh(action)
        
        # 计算log_prob
        log_prob = -0.5 * (((mean + std * noise - mean) / std) ** 2 + 2 * log_std + np.log(2 * np.pi))
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob -= torch.sum(torch.log(1 - action ** 2 + 1e-6), dim=-1, keepdim=True)
        
        return action, log_prob
    
    def log_prob(self, obs: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        state = obs["state"]
        features = self.net(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        # 逆tanh
        action = torch.clamp(action, -0.999, 0.999)
        pre_tanh = 0.5 * torch.log((1 + action) / (1 - action))
        
        log_prob = -0.5 * (((pre_tanh - mean) / std) ** 2 + 2 * log_std + np.log(2 * np.pi))
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob -= torch.sum(torch.log(1 - action ** 2 + 1e-6), dim=-1, keepdim=True)
        
        return log_prob
    
    def act(self, obs: Dict[str, Any], deterministic: bool = False) -> Any:
        with torch.no_grad():
            obs_tensor = {k: torch.as_tensor(v, dtype=torch.float32).unsqueeze(0).to(self.device) 
                         for k, v in obs.items() if isinstance(v, (np.ndarray, list))}
            
            if deterministic:
                action = self.forward(obs_tensor)
                action = torch.tanh(action)
            else:
                action, _ = self.sample(obs_tensor)
            
            return action.squeeze(0).cpu().numpy()
