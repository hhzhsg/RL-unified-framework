"""BC算法"""
from typing import Dict, Any
import torch
import torch.nn.functional as F
from .base_algorithm import BaseAlgorithm
from core.orchestration import register_algorithm


@register_algorithm("bc")
class BCAlgorithm(BaseAlgorithm):
    """Behavior Cloning"""
    
    def __init__(self, policy, config: Dict[str, Any]):
        super().__init__(config)
        self.policy = policy.to(self.device)
        self.lr = config.get("learning_rate", config.get("lr", 3e-4))
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
    
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        obs = {"state": torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)}
        action = torch.as_tensor(batch["action"], dtype=torch.float32, device=self.device)
        
        pred_action = self.policy(obs)
        pred_action = torch.tanh(pred_action)
        
        loss = F.mse_loss(pred_action, action)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self._train_step += 1
        return {"bc_loss": loss.item()}
    
    def get_policy(self):
        return self.policy
    
    def save(self, path: str) -> None:
        torch.save({"policy": self.policy.state_dict(), "train_step": self._train_step}, path)
    
    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self._train_step = ckpt["train_step"]
