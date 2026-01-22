"""均匀采样器"""
from typing import Dict, Any
import numpy as np
from .base_sampler import BaseSampler
from core.orchestration import register_sampler


@register_sampler("uniform")
class UniformSampler(BaseSampler):
    """均匀采样"""
    
    def sample(self, buffers: Dict[str, Any], batch_size: int) -> Dict[str, Any]:
        all_buffers = list(buffers.values())
        total_size = sum(len(b) for b in all_buffers)
        
        if total_size == 0:
            raise ValueError("All buffers are empty")
        
        samples_per_buffer = [int(batch_size * len(b) / total_size) for b in all_buffers]
        samples_per_buffer[-1] = batch_size - sum(samples_per_buffer[:-1])
        
        batches = []
        for buf, n in zip(all_buffers, samples_per_buffer):
            if n > 0 and len(buf) > 0:
                batches.append(buf.sample(n))
        
        if not batches:
            raise ValueError("No samples collected")
        
        result = {}
        for key in batches[0].keys():
            result[key] = np.concatenate([b[key] for b in batches], axis=0)
        return result
