"""
TD3+BC

Offline RL 算法，在 TD3 基础上添加 BC 正则化
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


class TD3BC(BaseAlgorithm):
    """
    TD3+BC
    
    Offline RL 算法: TD3 + Behavior Cloning 正则化
    
    要求 ModelGroup 包含:
    - policy: 确定性策略
    - q1, q2: QNetwork
    - target_policy: 目标策略 (frozen)
    - target_q1, target_q2: 目标 Q 网络 (frozen)
    """
    
    REQUIRED_MODELS = ["policy", "q1", "q2", "target_policy", "target_q1", "target_q2"]
    
    def __init__(self, model_group: ModelGroup, config: AlgorithmConfig = None):
        if config is None:
            config = AlgorithmConfig(name="td3bc", lr=3e-4)
        super().__init__(model_group, config)
        
        # 获取模型
        self.policy = model_group.get("policy")
        self.q1 = model_group.get("q1")
        self.q2 = model_group.get("q2")
        self.target_policy = model_group.get("target_policy")
        self.target_q1 = model_group.get("target_q1")
        self.target_q2 = model_group.get("target_q2")
        
        # 超参数
        self.gamma = getattr(config, 'gamma', 0.99)
        self.tau = getattr(config, 'tau', 0.005)
        self.policy_noise = getattr(config, 'policy_noise', 0.2)
        self.noise_clip = getattr(config, 'noise_clip', 0.5)
        self.policy_freq = getattr(config, 'policy_freq', 2)
        self.bc_alpha = getattr(config, 'bc_alpha', 2.5)  # BC 正则化系数
        
        # 优化器
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=config.lr
        )
        self.q_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=config.lr
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
            # 目标动作 + 噪声
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.target_policy.forward({}, next_state) + noise).clamp(-1, 1)
            
            # Target Q
            target_q1 = self.target_q1(next_state, next_action)
            target_q2 = self.target_q2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + self.gamma * (1 - done) * target_q
        
        # 当前 Q
        current_q1 = self.q1(state, action)
        current_q2 = self.q2(state, action)
        
        # Q 损失
        q_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # ========== 2. 延迟更新 Policy ==========
        policy_loss = torch.tensor(0.0)
        bc_loss = torch.tensor(0.0)
        
        if self._train_step_count % self.policy_freq == 0:
            # 当前策略动作
            pi = self.policy.forward({}, state)
            
            # Q 值
            q_value = self.q1(state, pi)
            
            # 归一化 Lambda (TD3+BC 关键)
            lam = self.bc_alpha / q_value.abs().mean().detach()
            
            # Policy 损失 = -Q + BC 正则
            bc_loss = F.mse_loss(pi, action)
            policy_loss = -lam * q_value.mean() + bc_loss
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            # 软更新目标网络
            self._soft_update(self.policy, self.target_policy)
            self._soft_update(self.q1, self.target_q1)
            self._soft_update(self.q2, self.target_q2)
        
        self._train_step_count += 1
        
        return {
            "q_loss": q_loss.item(),
            "policy_loss": policy_loss.item(),
            "bc_loss": bc_loss.item(),
            "q1": current_q1.mean().item(),
        }
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """软更新目标网络"""
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.copy_(
                self.tau * src_param.data + (1 - self.tau) * tgt_param.data
            )
