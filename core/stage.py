"""
VLA-RL 训练阶段
"""
from dataclasses import dataclass
from typing import List, Optional

from config import StageConfig


@dataclass
class Stage:
    """
    训练阶段
    支持多阶段训练 (如 RECAP: 先训练 VF，再训练 Policy)
    """
    name: str
    algorithm_name: str
    max_steps: int
    active_models: List[str]
    sample_strategy: str
    sample_kwargs: dict
    
    # 运行时状态
    current_step: int = 0
    
    @classmethod
    def from_config(cls, config: StageConfig) -> "Stage":
        return cls(
            name=config.name,
            algorithm_name=config.algorithm,
            max_steps=config.max_steps,
            active_models=config.active_models,
            sample_strategy=config.sample_strategy,
            sample_kwargs=config.sample_kwargs,
            current_step=0,
        )
    
    @property
    def is_finished(self) -> bool:
        return self.current_step >= self.max_steps
    
    @property
    def progress(self) -> float:
        return self.current_step / self.max_steps if self.max_steps > 0 else 1.0
