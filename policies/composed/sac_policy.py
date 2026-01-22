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

        # Optional discrete critic (for grippers / hybrid policies)
        self.num_discrete_actions = config.get("num_discrete_actions", None)
        if self.num_discrete_actions is not None:
            from policies.components.critics.q_critic import DiscreteCritic

            # discrete critic and its target
            dcfg = dict(config)
            dcfg["num_discrete_actions"] = self.num_discrete_actions
            self.discrete_critic = DiscreteCritic(dcfg)
            self.discrete_critic_target = copy.deepcopy(self.discrete_critic)
            for p in self.discrete_critic_target.parameters():
                p.requires_grad = False
    
    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()
    
    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.actor(obs)
    
    def act(self, obs: Dict[str, Any], deterministic: bool = False) -> Any:
        return self.actor.act(obs, deterministic)
    
    def sample(self, obs: Dict[str, torch.Tensor]):
        # Returns continuous action and log_prob; if discrete critic exists, append discrete action (argmax)
        action_cont, log_prob = self.actor.sample(obs)
        if self.num_discrete_actions is not None:
            # compute discrete Q and pick argmax as discrete action
            with torch.no_grad():
                qvals = self.discrete_critic(obs)  # [B, num_actions]
                discrete_action = torch.argmax(qvals, dim=-1, keepdim=True).float()
            action = torch.cat([action_cont, discrete_action], dim=-1)
            return action, log_prob
        return action_cont, log_prob
    
    def get_q_values(self, obs: Dict[str, torch.Tensor], action: torch.Tensor):
        return self.critics(obs, action)
    
    def get_target_q_values(self, obs: Dict[str, torch.Tensor], action: torch.Tensor):
        return self.target_critics(obs, action)

    def get_discrete_q(self, obs: Dict[str, torch.Tensor], use_target: bool = False) -> torch.Tensor:
        if self.num_discrete_actions is None:
            raise RuntimeError("Discrete critic not configured")
        if use_target:
            return self.discrete_critic_target(obs)
        return self.discrete_critic(obs)
    
    def update_target(self, tau: float = 0.005):
        for p, tp in zip(self.critics.parameters(), self.target_critics.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

        if self.num_discrete_actions is not None:
            for p, tp in zip(self.discrete_critic.parameters(), self.discrete_critic_target.parameters()):
                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
    
    def reset(self):
        pass
