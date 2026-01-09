"""
VLA-RL TD3+BC (Offline RL)

TD3+BC = TD3 (Twin Delayed DDPG) + Behavior Cloning 正则

核心思想:
- 用 TD 学习训练 Q 网络，估计动作价值
- 用 Q 值指导 Policy 更新，但加 BC loss 防止偏离数据分布

参考论文: A Minimalist Approach to Offline Reinforcement Learning
"""
from typing import Dict
import torch
import torch.optim as optim
import torch.nn.functional as F

from .base_algorithm import BaseAlgorithm
from model import ModelGroup
from data import Batch
from config import AlgorithmConfig


class TD3BC(BaseAlgorithm):
    """
    TD3 + Behavior Cloning
    
    要求 ModelGroup 包含:
    - policy: MLPPolicy
    - q1, q2: QNetwork
    - target_q1, target_q2: QNetwork (frozen)
    
    Loss 组成:
    1. Q Loss: TD 误差 (让 Q 网络准确估计动作价值)
    2. Policy Loss: -Q(s, π(s)) + α * BC_loss (最大化 Q 值，同时不偏离专家动作)
    """
    
    # 声明该算法需要的模型
    REQUIRED_MODELS = ["policy", "q1", "q2", "target_q1", "target_q2"]
    
    def __init__(self, model_group: ModelGroup, config: AlgorithmConfig = None):
        if config is None:
            config = AlgorithmConfig(name="td3_bc", lr=3e-4)
        super().__init__(model_group, config)
        
        # 验证 model_group
        self._validate_model_group()
        
        # 获取模型
        self.policy = model_group.get("policy")
        self.q1 = model_group.get("q1")
        self.q2 = model_group.get("q2")
        self.target_q1 = model_group.get("target_q1")
        self.target_q2 = model_group.get("target_q2")
        
        # 超参数 (支持从 algo_kwargs 获取)
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.bc_alpha = config.get('bc_alpha', 2.5)           # BC 正则系数
        self.policy_noise = config.get('policy_noise', 0.2)   # 目标动作噪声
        self.noise_clip = config.get('noise_clip', 0.5)       # 噪声裁剪
        self.policy_freq = config.get('policy_freq', 2)       # 延迟更新 policy
        
        # 优化器
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=config.lr)
        self.q_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=config.lr
        )
        
        # 设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _validate_model_group(self):
        """验证 model_group 包含所需模型"""
        missing = [name for name in self.REQUIRED_MODELS if name not in self.model_group]
        if missing:
            raise ValueError(
                f"TD3BC requires models {self.REQUIRED_MODELS}, "
                f"but missing: {missing}. "
                f"Available: {self.model_group.model_names}"
            )
    
    def train_step(self, batch: Batch) -> Dict[str, float]:
        """训练一步"""
        self.model_group.train(["policy", "q1", "q2"])
        
        # 转换数据
        batch = batch.to(self.device)
        state = batch.robot_state
        action = batch.action
        reward = batch.reward
        next_state = batch.next_robot_state
        done = batch.done
        
        # ==================== 1. 更新 Q 网络 ====================
        with torch.no_grad():
            # 目标动作 (加噪声，用于 target smoothing)
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = self.policy.forward({}, next_state)
            next_action = (next_action + noise).clamp(-1, 1)
            
            # 目标 Q 值 (取两个 Q 网络的较小值，减少过估计)
            target_q1 = self.target_q1(next_state, next_action)
            target_q2 = self.target_q2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * target_q
        
        # 当前 Q 值
        current_q1 = self.q1(state, action)
        current_q2 = self.q2(state, action)
        
        # Q Loss (TD 误差)
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        q_loss = q1_loss + q2_loss
        
        # 更新 Q 网络
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()), 1.0
        )
        self.q_optimizer.step()
        
        # ==================== 2. 延迟更新 Policy ====================
        policy_loss_val = 0.0
        bc_loss_val = 0.0
        
        if self._train_step_count % self.policy_freq == 0:
            # 预测动作
            pred_action = self.policy.forward({}, state)
            
            # Q 值 (用 Q1)
            q_value = self.q1(state, pred_action)
            
            # BC Loss (让预测动作接近专家动作)
            bc_loss = F.mse_loss(pred_action, action)
            
            # 归一化系数 λ = α / |Q|.mean()
            # 这让 Q 项和 BC 项在相似的量级
            lam = self.bc_alpha / (q_value.abs().mean().detach() + 1e-8)
            
            # Policy Loss = -Q + λ * BC
            policy_loss = -q_value.mean() + lam * bc_loss
            
            # 更新 Policy
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.policy_optimizer.step()
            
            policy_loss_val = policy_loss.item()
            bc_loss_val = bc_loss.item()
            
            # ==================== 3. 软更新 Target 网络 ====================
            self._soft_update()
        
        self._train_step_count += 1
        
        return {
            "q_loss": q_loss.item(),
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss_val,
            "bc_loss": bc_loss_val,
            "q_mean": current_q1.mean().item(),
            "train_step": self._train_step_count,
        }
    
    def _soft_update(self):
        """软更新 target 网络: target = τ * online + (1-τ) * target"""
        for param, target_param in zip(self.q1.parameters(), self.target_q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.q2.parameters(), self.target_q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
