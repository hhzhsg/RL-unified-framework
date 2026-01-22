"""HIL-SERL加权采样器"""
from typing import Dict, Any
import numpy as np
from .base_sampler import BaseSampler
from core.orchestration import register_sampler


@register_sampler("hilserl")
class HILSERLSampler(BaseSampler):
    """
    HIL-SERL加权采样器
    
    按权重从不同数据源采样:
    - demo: 离线演示数据
    - rollout: 在线rollout数据
    - intervention: 人类干预数据（2倍权重）
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            "demo": 1.0,
            "rollout": 1.0,
            "intervention": 2.0,
        }
    
    def sample(self, buffers: Dict[str, Any], batch_size: int) -> Dict[str, Any]:
        # 计算有效buffer和权重
        valid_buffers = {k: v for k, v in buffers.items() if len(v) > 0}
        if not valid_buffers:
            raise ValueError("All buffers are empty")
        
        # 归一化权重
        total_weight = sum(self.weights.get(k, 1.0) for k in valid_buffers)
        
        # 按权重分配样本数
        samples_per_buffer = {}
        remaining = batch_size
        for i, (name, buf) in enumerate(valid_buffers.items()):
            w = self.weights.get(name, 1.0)
            if i == len(valid_buffers) - 1:
                n = remaining
            else:
                n = int(batch_size * w / total_weight)
            samples_per_buffer[name] = min(n, len(buf))
            remaining -= samples_per_buffer[name]
        
        # 采样
        batches = []
        for name, n in samples_per_buffer.items():
            if n > 0:
                batches.append(buffers[name].sample(n))
        
        # 合并
        result = {}
        for key in batches[0].keys():
            result[key] = np.concatenate([b[key] for b in batches], axis=0)
        
        # 打乱
        indices = np.random.permutation(len(result[list(result.keys())[0]]))
        return {k: v[indices] for k, v in result.items()}
    
    def set_weights(self, weights: Dict[str, float]) -> None:
        self.weights.update(weights)
