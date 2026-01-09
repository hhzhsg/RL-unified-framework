"""
VLA-RL Soft Actor-Critic (SAC)
"""
from typing import Dict
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .base_algorithm import BaseAlgorithm
from model import ModelGroup
from data import Batch
from config import AlgorithmConfig


class QNetwork(nn.Module):
    """Q 网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims=[256, 256]):
        super().__init__()
        
        layers = []
        input_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class SAC(BaseAlgorithm):
    """
    Soft Actor-Critic
    在线强化学习算法
    """
    
    def __init__(self, model_group: ModelGroup, config: AlgorithmConfig = None,
                 state_dim: int = 14, action_dim: int = 7):
        if config is None:
            config = AlgorithmConfig(name="sac", lr=3e-4)
        super().__init__(model_group, config)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = config.alpha
        self.auto_alpha = config.auto_alpha
        
        # 确保 model_group 包含所需模型
        if "critic1" not in model_group:
            model_group.add("critic1", QNetwork(state_dim, action_dim))
        if "critic2" not in model_group:
            model_group.add("critic2", QNetwork(state_dim, action_dim))
        if "critic1_target" not in model_group:
            model_group.add("critic1_target", copy.deepcopy(model_group.get("critic1")))
        if "critic2_target" not in model_group:
            model_group.add("critic2_target", copy.deepcopy(model_group.get("critic2")))
        
        # 优化器
        self.actor_optimizer = optim.Adam(
            model_group.trainable_parameters(["policy"]),
            lr=config.lr
        )
        self.critic1_optimizer = optim.Adam(
            model_group.get("critic1").parameters(),
            lr=config.lr
        )
        self.critic2_optimizer = optim.Adam(
            model_group.get("critic2").parameters(),
            lr=config.lr
        )
        
        # 自动温度调节
        if self.auto_alpha:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.lr)
    
    def train_step(self, batch: Batch) -> Dict[str, float]:
        """训练一步"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch = batch.to(device)
        
        # 更新 Critic
        critic_loss = self._update_critic(batch)
        
        # 更新 Actor
        actor_loss, alpha_loss = self._update_actor(batch)
        
        # 软更新 target
        self._soft_update()
        
        self._train_step_count += 1
        
        return {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": self.alpha,
            "train_step": self._train_step_count,
        }
    
    def _update_critic(self, batch: Batch) -> float:
        """更新 Critic"""
        policy = self.model_group.get("policy")
        critic1 = self.model_group.get("critic1")
        critic2 = self.model_group.get("critic2")
        critic1_target = self.model_group.get("critic1_target")
        critic2_target = self.model_group.get("critic2_target")
        
        with torch.no_grad():
            # 采样下一动作
            next_action, next_log_prob = policy.sample(batch.next_obs, batch.next_robot_state)
            
            # 计算 target Q
            target_q1 = critic1_target(batch.next_robot_state, next_action)
            target_q2 = critic2_target(batch.next_robot_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = batch.reward.unsqueeze(-1) + self.gamma * (1 - batch.done.unsqueeze(-1)) * target_q
        
        # 计算当前 Q
        current_q1 = critic1(batch.robot_state, batch.action)
        current_q2 = critic2(batch.robot_state, batch.action)
        
        # 损失
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        # 更新
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        return (critic1_loss.item() + critic2_loss.item()) / 2
    
    def _update_actor(self, batch: Batch):
        """更新 Actor"""
        policy = self.model_group.get("policy")
        critic1 = self.model_group.get("critic1")
        critic2 = self.model_group.get("critic2")
        
        action, log_prob = policy.sample(batch.obs, batch.robot_state)
        
        q1 = critic1(batch.robot_state, action)
        q2 = critic2(batch.robot_state, action)
        q = torch.min(q1, q2)
        
        actor_loss = (self.alpha * log_prob - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 更新 alpha
        alpha_loss = 0.0
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
            alpha_loss = alpha_loss.item()
        
        return actor_loss.item(), alpha_loss
    
    def _soft_update(self):
        """软更新 target 网络"""
        critic1 = self.model_group.get("critic1")
        critic2 = self.model_group.get("critic2")
        critic1_target = self.model_group.get("critic1_target")
        critic2_target = self.model_group.get("critic2_target")
        
        for param, target_param in zip(critic1.parameters(), critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(critic2.parameters(), critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
