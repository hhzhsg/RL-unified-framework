"""
Soft Actor-Critic (SAC)

Online RL 算法，支持自动温度调节
"""
from typing import Dict
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .base import BaseAlgorithm
from data import Batch
from core import ModelGroup
from config import AlgorithmConfig


class SAC(BaseAlgorithm):
    """
    Soft Actor-Critic
    
    要求 ModelGroup 包含:
    - policy: MLPGaussianPolicy (必须有 sample 方法)
    - q1, q2: QNetwork
    - target_q1, target_q2: QNetwork (frozen)
    """
    
    REQUIRED_MODELS = ["policy", "q1", "q2", "target_q1", "target_q2"]
    
    def __init__(self, model_group: ModelGroup, config: AlgorithmConfig = None):
        if config is None:
            config = AlgorithmConfig(name="sac", lr=3e-4)
        super().__init__(model_group, config)
        
        # 获取模型引用
        self.policy = model_group.get("policy")
        self.q1 = model_group.get("q1")
        self.q2 = model_group.get("q2")
        self.target_q1 = model_group.get("target_q1")
        self.target_q2 = model_group.get("target_q2")
        
        # 超参数
        self.gamma = getattr(config, 'gamma', 0.99)
        self.tau = getattr(config, 'tau', 0.005)
        self.alpha = getattr(config, 'alpha', 0.2)
        self.auto_alpha = getattr(config, 'auto_alpha', True)
        
        # 设备 (先获取，后面要用)
        self.device = next(self.policy.parameters()).device
        
        # 优化器
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=config.lr
        )
        self.q_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=config.lr
        )
        
        # 自动温度调节
        if self.auto_alpha:
            action_dim = self.policy.action_dim
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.lr)
    
    def _validate_model_group(self):
        """验证 model_group"""
        super()._validate_model_group()
        
        # 验证 policy 有 sample 方法
        policy = self.model_group.get("policy")
        if not hasattr(policy, 'sample'):
            raise ValueError(
                "SAC requires policy with sample() method. "
                "Use MLPGaussianPolicy instead of MLPPolicy."
            )
    
    def train_step(self, batch: Batch) -> Dict[str, float]:
        """训练一步"""
        state = batch.robot_state
        action = batch.action
        reward = batch.reward.unsqueeze(-1)
        next_state = batch.next_robot_state
        done = batch.done.unsqueeze(-1)
        
        # ========== 1. 更新 Q 网络 ==========
        with torch.no_grad():
            # 采样下一步动作
            next_action, next_log_prob = self.policy.sample(next_state)
            
            # 计算 target Q
            target_q1 = self.target_q1(next_state, next_action)
            target_q2 = self.target_q2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + self.gamma * (1 - done) * target_q
        
        # 当前 Q
        current_q1 = self.q1(state, action)
        current_q2 = self.q2(state, action)
        
        # Q 损失
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        q_loss = q1_loss + q2_loss
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # ========== 2. 更新 Policy ==========
        new_action, log_prob = self.policy.sample(state)
        q1_new = self.q1(state, new_action)
        q2_new = self.q2(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.alpha * log_prob - q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # ========== 3. 更新 Alpha (可选) ==========
        alpha_loss = 0.0
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
        
        # ========== 4. 软更新 Target ==========
        self._soft_update(self.q1, self.target_q1)
        self._soft_update(self.q2, self.target_q2)
        
        self._train_step_count += 1
        
        return {
            "q_loss": q_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha": self.alpha,
            "alpha_loss": alpha_loss.item() if self.auto_alpha else 0.0,
            "q1": current_q1.mean().item(),
            "q2": current_q2.mean().item(),
        }
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """软更新目标网络"""
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.copy_(
                self.tau * src_param.data + (1 - self.tau) * tgt_param.data
            )
