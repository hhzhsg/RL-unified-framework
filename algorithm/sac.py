"""
VLA-RL Soft Actor-Critic (SAC)

Online RL 算法，支持自动温度调节
"""
from typing import Dict
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .base_algorithm import BaseAlgorithm
from model import ModelGroup
from model.q_network import QNetwork  # 统一使用 model/ 下的 QNetwork
from data import Batch
from config import AlgorithmConfig


class SAC(BaseAlgorithm):
    """
    Soft Actor-Critic
    
    要求 ModelGroup 包含:
    - policy: MLPGaussianPolicy (必须有 sample 方法)
    - q1, q2: QNetwork
    - target_q1, target_q2: QNetwork (frozen)
    """
    
    # 声明该算法需要的模型
    REQUIRED_MODELS = ["policy", "q1", "q2", "target_q1", "target_q2"]
    
    def __init__(self, model_group: ModelGroup, config: AlgorithmConfig = None):
        if config is None:
            config = AlgorithmConfig(name="sac", lr=3e-4)
        super().__init__(model_group, config)
        
        # 验证 model_group 包含所需模型
        self._validate_model_group()
        
        # 获取模型引用
        self.policy = model_group.get("policy")
        self.q1 = model_group.get("q1")
        self.q2 = model_group.get("q2")
        self.target_q1 = model_group.get("target_q1")
        self.target_q2 = model_group.get("target_q2")
        
        # 超参数 (从 config 或 algo_kwargs 获取)
        self.gamma = getattr(config, 'gamma', 0.99)
        self.tau = getattr(config, 'tau', 0.005)
        self.alpha = getattr(config, 'alpha', 0.2)
        self.auto_alpha = getattr(config, 'auto_alpha', True)
        
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
            # 目标熵 = -action_dim
            action_dim = self.policy.action_dim
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.lr)
        
        # 设备
        self.device = next(self.policy.parameters()).device
    
    def _validate_model_group(self):
        """验证 model_group 包含所需模型"""
        missing = [name for name in self.REQUIRED_MODELS if name not in self.model_group]
        if missing:
            raise ValueError(
                f"SAC requires models {self.REQUIRED_MODELS}, "
                f"but missing: {missing}. "
                f"Available: {self.model_group.model_names}"
            )
        
        # 验证 policy 有 sample 方法
        policy = self.model_group.get("policy")
        if not hasattr(policy, 'sample'):
            raise ValueError(
                "SAC requires policy with sample() method. "
                "Use MLPGaussianPolicy instead of MLPPolicy."
            )
    
    def train_step(self, batch: Batch) -> Dict[str, float]:
        """训练一步"""
        self.model_group.train(["policy", "q1", "q2"])
        
        batch = batch.to(self.device)
        
        # 更新 Critic (Q 网络)
        q_loss, q_info = self._update_critic(batch)
        
        # 更新 Actor (Policy)
        policy_loss, alpha_loss = self._update_actor(batch)
        
        # 软更新 target 网络
        self._soft_update()
        
        self._train_step_count += 1
        
        return {
            "q_loss": q_loss,
            "policy_loss": policy_loss,
            "alpha_loss": alpha_loss,
            "alpha": self.alpha,
            "q_mean": q_info["q_mean"],
            "train_step": self._train_step_count,
        }
    
    def _update_critic(self, batch: Batch) -> tuple:
        """更新 Q 网络"""
        with torch.no_grad():
            # 采样下一状态的动作
            next_action, next_log_prob = self.policy.sample({}, batch.next_robot_state)
            
            # 计算 target Q 值 (取较小值，减少过估计)
            target_q1 = self.target_q1(batch.next_robot_state, next_action)
            target_q2 = self.target_q2(batch.next_robot_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob.squeeze(-1)
            
            # TD target
            target_q = batch.reward + (1 - batch.done) * self.gamma * target_q
        
        # 当前 Q 值
        current_q1 = self.q1(batch.robot_state, batch.action)
        current_q2 = self.q2(batch.robot_state, batch.action)
        
        # Q Loss
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        q_loss = q1_loss + q2_loss
        
        # 更新
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()), 1.0
        )
        self.q_optimizer.step()
        
        return q_loss.item(), {"q_mean": current_q1.mean().item()}
    
    def _update_actor(self, batch: Batch) -> tuple:
        """更新 Policy"""
        # 采样动作
        action, log_prob = self.policy.sample({}, batch.robot_state)
        
        # Q 值
        q1 = self.q1(batch.robot_state, action)
        q2 = self.q2(batch.robot_state, action)
        q = torch.min(q1, q2)
        
        # Policy Loss: 最大化 Q 值，同时最大化熵
        policy_loss = (self.alpha * log_prob.squeeze(-1) - q).mean()
        
        # 更新 Policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optimizer.step()
        
        # 更新 alpha (温度系数)
        alpha_loss = 0.0
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob.squeeze(-1) + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
            alpha_loss = alpha_loss.item()
        
        return policy_loss.item(), alpha_loss
    
    def _soft_update(self):
        """软更新 target 网络: target = τ * online + (1-τ) * target"""
        for param, target_param in zip(self.q1.parameters(), self.target_q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.q2.parameters(), self.target_q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
