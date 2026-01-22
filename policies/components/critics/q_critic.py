"""Q-Critic"""
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from policies.base_policy import BasePolicy
from core.interfaces import CriticInterface
from core.orchestration import register_policy


@register_policy("q_critic")
class QCritic(BasePolicy, CriticInterface):
    """Q-Critic"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        hidden_dims = config.get("hidden_dims", [256, 256])
        
        layers = []
        prev_dim = self.state_dim + self.action_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, obs: Dict[str, torch.Tensor], action: Optional[torch.Tensor] = None) -> torch.Tensor:
        state = obs["state"]
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
    
    def act(self, obs: Dict[str, Any], deterministic: bool = False) -> Any:
        raise NotImplementedError("Critic does not have act()")


@register_policy("q_critic_ensemble")
class QCriticEnsemble(BasePolicy):
    """Q-Critic集成（用于SAC）"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.num_critics = config.get("num_critics", 2)
        self.critics = nn.ModuleList([
            QCritic(config) for _ in range(self.num_critics)
        ])
    
    def forward(self, obs: Dict[str, torch.Tensor], action: torch.Tensor) -> List[torch.Tensor]:
        return [critic(obs, action) for critic in self.critics]
    
    def act(self, obs: Dict[str, Any], deterministic: bool = False) -> Any:
        raise NotImplementedError("Critic does not have act()")


@register_policy("discrete_critic")
class DiscreteCritic(BasePolicy):
    """离散动作的 Critic（用于 gripper / discrete action Q）"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.state_dim = config.get("state_dim")
        # input_dim is observation encoding only
        hidden_dims = config.get("discrete_hidden_dims", [128, 128])
        output_dim = config.get("num_discrete_actions", 3)

        layers = []
        prev_dim = self.state_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        state = obs["state"]
        return self.net(state)

    def act(self, obs: Dict[str, Any], deterministic: bool = False) -> Any:
        with torch.no_grad():
            q = self.forward(obs)
            return torch.argmax(q, dim=-1)
