"""SAC策略组合"""
from typing import Dict, Any
import torch
import copy
from policies.base_policy import BasePolicy
from ..components.actors import GaussianActor
from ..components.critics import QCriticEnsemble
from core.orchestration import register_policy


@register_policy("sac")
class SACPolicy(BasePolicy):
    """
    SAC策略组合
    
    包含:
    - Actor (GaussianActor)
    - Critic Ensemble
    - Target Critic Ensemble
    - Temperature
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Actor
        self.actor = GaussianActor(config)
        
        # Critics
        self.critics = QCriticEnsemble(config)
        self.target_critics = copy.deepcopy(self.critics)
        
        # Freeze target
        for p in self.target_critics.parameters():
            p.requires_grad = False
        
        # Temperature
        init_temp = config.get("init_temperature", 0.1)
        self.log_alpha = torch.nn.Parameter(torch.tensor(init_temp).log())
        
        # Target entropy
        self.target_entropy = -config["action_dim"]
    
    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()
    
    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.actor(obs)
    
    def act(self, obs: Dict[str, Any], deterministic: bool = False) -> Any:
        return self.actor.act(obs, deterministic)
    
    def sample(self, obs: Dict[str, torch.Tensor]):
        return self.actor.sample(obs)
    
    def get_q_values(self, obs: Dict[str, torch.Tensor], action: torch.Tensor):
        return self.critics(obs, action)
    
    def get_target_q_values(self, obs: Dict[str, torch.Tensor], action: torch.Tensor):
        return self.target_critics(obs, action)
    
    def update_target(self, tau: float = 0.005):
        for p, tp in zip(self.critics.parameters(), self.target_critics.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
    
    def reset(self):
        pass
