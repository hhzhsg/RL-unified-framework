"""奖励分类器"""
from typing import Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from policies.components.encoders.image_encoder import ImageEncoder


class RewardClassifier(nn.Module):
    """
    基于视觉的奖励分类器
    
    用于HIL-SERL自动检测任务成功/失败
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.num_cameras = config.get("num_cameras", 2)
        self.num_classes = config.get("num_classes", 2)
        self.hidden_dim = config.get("hidden_dim", 256)
        input_shape = config.get("input_shape", (3, 128, 128))
        
        # 每个相机一个编码器
        self.encoders = nn.ModuleList([
            ImageEncoder(input_shape, self.hidden_dim)
            for _ in range(self.num_cameras)
        ])
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * self.num_cameras, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.hidden_dim, self.num_classes),
        )
    
    def forward(self, images: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            images: 相机图像列表 [Tensor(B,C,H,W), ...]
        Returns:
            logits: (B, num_classes)
        """
        features = []
        for img, encoder in zip(images, self.encoders):
            features.append(encoder(img))
        
        combined = torch.cat(features, dim=-1)
        return self.classifier(combined)
    
    def predict_reward(self, images: List[torch.Tensor], threshold: float = 0.5) -> torch.Tensor:
        """预测奖励"""
        with torch.no_grad():
            logits = self.forward(images)
            probs = F.softmax(logits, dim=-1)
            success_prob = probs[:, 1]  # class 1 = success
            return (success_prob > threshold).float()
    
    @classmethod
    def load_pretrained(cls, path: str, config: Dict[str, Any]) -> "RewardClassifier":
        model = cls(config)
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model
