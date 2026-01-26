"""
Simple MLP 适配器

用于框架测试和调试，不依赖完整的 Algorithm/Policy 实现。
提供最简单的 PolicyAdapter 和 TrainableAdapter 实现。

使用场景：
- HIL 通信测试
- gRPC 压力测试
- 框架集成测试
"""
from typing import Dict, Any, Optional
import os
import numpy as np
import torch
import torch.nn as nn


class SimpleMLPPolicy(nn.Module):
    """简单的 MLP 策略网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleMLPAdapter:
    """
    简单 MLP 策略适配器（符合 PolicyAdapterProtocol）
    
    用于 Actor 端推理和权重同步测试
    
    使用示例:
        adapter = SimpleMLPAdapter(state_dim=37, action_dim=23)
        action = adapter.act({"state": obs})
        weights = adapter.get_weights()
        adapter.load_weights(new_weights)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        self.policy = SimpleMLPPolicy(state_dim, action_dim, hidden_dim).to(device)
        self._weight_version = 0
    
    def act(self, obs: Dict[str, Any], deterministic: bool = False) -> np.ndarray:
        """推理动作"""
        # 提取状态
        if isinstance(obs, dict):
            state = obs.get("state", obs.get("observation", obs.get("proprio")))
        else:
            state = obs
        
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        
        with torch.no_grad():
            action = self.policy(state_tensor)
        
        return action.cpu().numpy().squeeze(0)
    
    def get_weights(self) -> Dict[str, torch.Tensor]:
        """获取权重（用于同步）"""
        return {k: v.cpu() for k, v in self.policy.state_dict().items()}
    
    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """加载权重"""
        self.policy.load_state_dict(weights)
        self._weight_version += 1
    
    def reset(self) -> None:
        """Episode 重置（无状态策略不需要）"""
        pass


class SimpleMLPTrainer:
    """
    简单 MLP 训练器（符合 TrainableAdapterProtocol）
    
    用于 Learner 端训练和权重发布测试
    
    使用示例:
        trainer = SimpleMLPTrainer(state_dim=37, action_dim=23, device="cuda")
        metrics = trainer.update(batch)
        weights = trainer.get_weights()
        trainer.save("checkpoint.pt")
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        self.policy = SimpleMLPPolicy(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        self._update_count = 0
    
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """训练一步（行为克隆）"""
        obs = batch["obs"]
        action = batch["action"]
        
        # 转换为 tensor
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if isinstance(action, np.ndarray):
            action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        
        # 前向传播
        pred_action = self.policy(obs)
        loss = self.loss_fn(pred_action, action)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self._update_count += 1
        
        return {"loss": loss.item()}
    
    def get_weights(self) -> Dict[str, torch.Tensor]:
        """获取权重（用于发布）"""
        return {k: v.cpu() for k, v in self.policy.state_dict().items()}
    
    def save(self, path: str) -> None:
        """保存 checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update_count": self._update_count,
        }, path)
    
    def load(self, path: str) -> None:
        """加载 checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._update_count = checkpoint.get("update_count", 0)
