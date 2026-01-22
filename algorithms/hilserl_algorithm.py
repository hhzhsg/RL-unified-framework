"""HIL-SERL算法"""
from typing import Dict, Any
from .sac_algorithm import SACAlgorithm
from policies.composed import HILSERLPolicy
from core.orchestration import register_algorithm


@register_algorithm("hilserl")
class HILSERLAlgorithm(SACAlgorithm):
    """
    HIL-SERL算法
    
    继承SAC，扩展:
    - 支持干预数据的特殊处理
    - UTD ratio (Update-to-Data ratio)
    """
    
    def __init__(self, policy: HILSERLPolicy, config: Dict[str, Any]):
        super().__init__(policy, config)
        
        self.utd_ratio = config.get("utd_ratio", 1)
        self.intervention_bonus = config.get("intervention_bonus", 0.0)
    
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        # 干预数据奖励bonus
        if self.intervention_bonus > 0 and "source" in batch:
            sources = batch.get("source", [])
            for i, src in enumerate(sources):
                if src == "intervention":
                    batch["reward"][i] += self.intervention_bonus
        
        # 调用父类更新
        return super().update(batch)
    
    def update_multiple(self, batch: Dict[str, Any], utd: int = None) -> Dict[str, float]:
        """多次更新（UTD ratio）"""
        utd = utd or self.utd_ratio
        metrics_sum = {}
        
        for _ in range(utd):
            metrics = self.update(batch)
            for k, v in metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0) + v
        
        return {k: v / utd for k, v in metrics_sum.items()}
