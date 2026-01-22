"""SAC算法"""
from typing import Dict, Any
import torch
import torch.nn.functional as F
from .base_algorithm import BaseOffPolicyAlgorithm
from policies.composed import SACPolicy
from core.orchestration import register_algorithm


@register_algorithm("sac")
class SACAlgorithm(BaseOffPolicyAlgorithm):
    """
    Soft Actor-Critic算法
    """
    
    def __init__(self, policy: SACPolicy, config: Dict[str, Any]):
        super().__init__(config)
        
        self.policy = policy.to(self.device)
        
        # 超参数
        self.gamma = config.get("gamma", 0.99)
        self.actor_lr = config.get("actor_lr", 3e-4)
        self.critic_lr = config.get("critic_lr", 3e-4)
        self.alpha_lr = config.get("alpha_lr", 3e-4)
        self.num_critics = config.get("num_critics", 2)
        self.num_subsample = config.get("num_subsample_critics", 2)
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(), lr=self.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.policy.critics.parameters(), lr=self.critic_lr
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.policy.log_alpha], lr=self.alpha_lr
        )
    
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        # 转换为tensor
        obs = {"state": torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)}
        action = torch.as_tensor(batch["action"], dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(batch["reward"], dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_obs = {"state": torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)}
        done = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device).unsqueeze(-1)
        
        # 1. Critic更新
        with torch.no_grad():
            next_action, next_log_prob = self.policy.sample(next_obs)
            target_qs = self.policy.get_target_q_values(next_obs, next_action)
            target_q = torch.min(torch.stack(target_qs), dim=0)[0]
            target_q = target_q - self.policy.alpha.detach() * next_log_prob
            target_value = reward + (1 - done) * self.gamma * target_q
        
        current_qs = self.policy.get_q_values(obs, action)
        critic_loss = sum(F.mse_loss(q, target_value) for q in current_qs)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 2. Actor更新
        new_action, log_prob = self.policy.sample(obs)
        qs = self.policy.get_q_values(obs, new_action)
        min_q = torch.min(torch.stack(qs), dim=0)[0]
        actor_loss = (self.policy.alpha.detach() * log_prob - min_q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 3. Temperature更新
        alpha_loss = -(self.policy.log_alpha * (log_prob + self.policy.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # 4. 更新目标网络
        self.update_target()
        
        self._train_step += 1
        
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.policy.alpha.item(),
            "q_value": min_q.mean().item(),
        }
    
    def update_target(self) -> None:
        self.policy.update_target(self.tau)
    
    def get_policy(self):
        return self.policy
    
    def save(self, path: str) -> None:
        torch.save({
            "policy": self.policy.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "train_step": self._train_step,
        }, path)
    
    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(ckpt["alpha_optimizer"])
        self._train_step = ckpt["train_step"]
