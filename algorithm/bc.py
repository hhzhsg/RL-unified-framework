"""
Behavior Cloning (BC)

监督学习的行为克隆算法
"""
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim

from .base import BaseAlgorithm
from data import Batch
from core import ModelGroup
from config import AlgorithmConfig


class BC(BaseAlgorithm):
    """
    Behavior Cloning
    
    最简单的离线模仿学习算法: 最小化专家动作的 MSE 损失
    
    要求 ModelGroup 包含:
    - policy: 策略网络
    """
    
    REQUIRED_MODELS = ["policy"]
    
    def __init__(self, model_group: ModelGroup, config: AlgorithmConfig = None):
        if config is None:
            config = AlgorithmConfig(name="bc", lr=1e-4)
        super().__init__(model_group, config)
        
        self.policy = model_group.get("policy")
        
        # 优化器
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=config.lr,
        )
        
        # 损失函数
        self.loss_fn = nn.MSELoss()
    
    def train_step(self, batch: Batch) -> Dict[str, float]:
        """训练一步"""
        self.policy.train()
        
        # 前向传播
        pred_action = self.policy.forward(batch.obs, batch.robot_state)
        
        # 计算损失
        loss = self.loss_fn(pred_action, batch.action)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪 (可选)
        if hasattr(self.config, 'grad_clip') and self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip)
        
        self.optimizer.step()
        
        self._train_step_count += 1
        
        return {
            "loss": loss.item(),
            "bc_loss": loss.item(),
        }
