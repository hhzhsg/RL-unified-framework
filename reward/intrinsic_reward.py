"""
Intrinsic Reward (内在奖励)

好奇心驱动的探索奖励：
- RND (Random Network Distillation)
- ICM (Intrinsic Curiosity Module) - 待实现
"""
from typing import Optional, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base_reward import BaseReward
from data import Batch


class RNDNetwork(nn.Module):
    """RND 网络"""
    
    def __init__(self, input_dim: int, output_dim: int = 64, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNDReward(BaseReward):
    """
    Random Network Distillation
    
    使用预测误差作为内在奖励：
    r_int = ||f(s) - f_target(s)||^2
    
    其中 f_target 是随机初始化后固定的网络，f 是要学习的预测网络。
    对于新颖的状态，预测误差大，内在奖励高。
    
    参考: Burda et al., "Exploration by Random Network Distillation" (2018)
    """
    
    def __init__(self,
                 state_dim: int,
                 feature_dim: int = 64,
                 hidden_dims: List[int] = [256, 256],
                 lr: float = 1e-4,
                 intrinsic_scale: float = 1.0,
                 extrinsic_scale: float = 1.0,
                 update_freq: int = 1,
                 device: str = "cpu"):
        """
        Args:
            state_dim: 状态维度
            feature_dim: 特征维度
            hidden_dims: 隐藏层维度
            lr: 学习率
            intrinsic_scale: 内在奖励系数
            extrinsic_scale: 外在奖励系数
            update_freq: 更新频率
            device: 计算设备
        """
        super().__init__(name="rnd")
        
        self.intrinsic_scale = intrinsic_scale
        self.extrinsic_scale = extrinsic_scale
        self.update_freq = update_freq
        self.device = device
        
        # 目标网络（随机初始化，固定）
        self.target_net = RNDNetwork(state_dim, feature_dim, hidden_dims).to(device)
        for p in self.target_net.parameters():
            p.requires_grad = False
        
        # 预测网络（要学习的）
        self.predictor_net = RNDNetwork(state_dim, feature_dim, hidden_dims).to(device)
        
        # 优化器
        self.optimizer = optim.Adam(self.predictor_net.parameters(), lr=lr)
        
        # 奖励归一化
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0
        
        # 更新计数
        self._update_count = 0
    
    def compute(self,
                state: np.ndarray,
                action: np.ndarray,
                next_state: np.ndarray,
                env_reward: float,
                done: bool,
                info: Optional[Dict[str, Any]] = None) -> float:
        """计算 RND 奖励"""
        # 计算内在奖励
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            target_feature = self.target_net(next_state_tensor)
            pred_feature = self.predictor_net(next_state_tensor)
            intrinsic = ((target_feature - pred_feature) ** 2).sum(dim=-1).item()
        
        # 归一化内在奖励
        intrinsic_normalized = intrinsic / (self.reward_std + 1e-8)
        
        # 组合奖励
        total_reward = (self.extrinsic_scale * env_reward + 
                       self.intrinsic_scale * intrinsic_normalized)
        
        return total_reward
    
    def update(self, batch: Batch):
        """更新预测网络"""
        self._update_count += 1
        if self._update_count % self.update_freq != 0:
            return
        
        # 获取 next_state
        if isinstance(batch.next_robot_state, torch.Tensor):
            next_states = batch.next_robot_state.to(self.device)
        else:
            next_states = torch.FloatTensor(batch.next_robot_state).to(self.device)
        
        # 计算预测误差
        with torch.no_grad():
            target_features = self.target_net(next_states)
        pred_features = self.predictor_net(next_states)
        
        loss = ((target_features - pred_features) ** 2).sum(dim=-1).mean()
        
        # 更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新奖励统计
        with torch.no_grad():
            intrinsic_rewards = ((target_features - pred_features) ** 2).sum(dim=-1)
            batch_mean = intrinsic_rewards.mean().item()
            batch_std = intrinsic_rewards.std().item()
            
            # 滑动平均更新
            alpha = 0.01
            self.reward_mean = (1 - alpha) * self.reward_mean + alpha * batch_mean
            self.reward_std = (1 - alpha) * self.reward_std + alpha * batch_std
    
    def get_stats(self) -> Dict[str, float]:
        return {
            "call_count": self._call_count,
            "update_count": self._update_count,
            "reward_mean": self.reward_mean,
            "reward_std": self.reward_std,
        }


class ICMReward(BaseReward):
    """
    Intrinsic Curiosity Module (预留实现)
    
    TODO: 实现 ICM
    - Forward model: 预测 s' given (s, a)
    - Inverse model: 预测 a given (s, s')
    - Intrinsic reward: forward model 预测误差
    """
    
    def __init__(self, state_dim: int, action_dim: int, **kwargs):
        super().__init__(name="icm")
        raise NotImplementedError("ICM not implemented yet")
    
    def compute(self, *args, **kwargs) -> float:
        raise NotImplementedError()
