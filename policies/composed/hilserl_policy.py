"""HIL-SERL策略"""
from typing import Dict, Any, Optional
import torch
from .sac_policy import SACPolicy
from ..components.classifiers import RewardClassifier
from core.orchestration import register_policy


@register_policy("hilserl")
class HILSERLPolicy(SACPolicy):
    """
    HIL-SERL策略
    
    继承SAC，添加:
    - 奖励分类器
    - 干预状态跟踪
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 奖励分类器（可选）
        self.reward_classifier: Optional[RewardClassifier] = None
        if config.get("reward_classifier"):
            rc_config = config["reward_classifier"]
            self.reward_classifier = RewardClassifier(rc_config)
            if rc_config.get("pretrained_path"):
                self.reward_classifier = RewardClassifier.load_pretrained(
                    rc_config["pretrained_path"], rc_config
                )
        
        # 干预状态
        self._is_intervening = False
    
    def set_intervention(self, is_intervening: bool):
        self._is_intervening = is_intervening
    
    @property
    def is_intervening(self) -> bool:
        return self._is_intervening
    
    def predict_reward(self, images: list, threshold: float = 0.5) -> float:
        """使用奖励分类器预测奖励"""
        if self.reward_classifier is None:
            return 0.0
        return self.reward_classifier.predict_reward(images, threshold).item()
