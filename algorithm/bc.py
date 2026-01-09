"""
VLA-RL Behavior Cloning (BC)
"""
from typing import Dict
import torch
import torch.optim as optim
import torch.nn.functional as F

from .base_algorithm import BaseAlgorithm
from model import ModelGroup
from data import Batch
from config import AlgorithmConfig


class BC(BaseAlgorithm):
    """
    Behavior Cloning
    监督学习模仿专家数据
    """
    
    def __init__(self, model_group: ModelGroup, config: AlgorithmConfig = None):
        if config is None:
            config = AlgorithmConfig(name="bc", lr=1e-4)
        super().__init__(model_group, config)
        
        # 优化器
        self.optimizer = optim.Adam(
            model_group.trainable_parameters(["policy"]),
            lr=config.lr
        )
    
    def train_step(self, batch: Batch) -> Dict[str, float]:
        """训练一步"""
        self.model_group.train(["policy"])
        
        # 转换为 tensor
        batch = batch.to("cuda" if torch.cuda.is_available() else "cpu")
        
        # 前向
        policy = self.model_group.get("policy")
        pred_action = policy.forward(batch.obs, batch.robot_state)
        
        # 损失
        loss = F.mse_loss(pred_action, batch.action)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        self.optimizer.step()
        
        self._train_step_count += 1
        
        return {
            "loss": loss.item(),
            "train_step": self._train_step_count,
        }
