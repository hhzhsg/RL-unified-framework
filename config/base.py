"""
配置定义

所有模块的配置 dataclass
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class EnvConfig:
    """环境配置"""
    name: str = "dummy"
    state_dim: int = 16
    action_dim: int = 4
    max_episode_steps: int = 200
    action_space: str = "joint"


@dataclass
class ModelConfig:
    """模型配置"""
    policy_type: str = "mlp"
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"


@dataclass
class AlgorithmConfig:
    """算法配置"""
    name: str = "bc"
    lr: float = 1e-4
    batch_size: int = 64
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    auto_alpha: bool = True
    grad_clip: float = 0.0
    # 额外参数
    algo_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """数据配置"""
    demo_paths: List[str] = field(default_factory=list)
    camera_keys: List[str] = field(default_factory=list)
    load_images: bool = False
    rollout_capacity: int = 100000


@dataclass
class TrainingConfig:
    """训练配置"""
    total_steps: int = 10000
    batch_size: int = 64
    log_freq: int = 100
    checkpoint_freq: int = 1000
    checkpoint_dir: str = "./checkpoints"


@dataclass
class InferenceConfig:
    """推理配置"""
    device: str = "cpu"
    deterministic: bool = False
    action_horizon: int = 1


@dataclass
class WeightSyncConfig:
    """权重同步配置"""
    method: str = "shared_memory"
    sync_freq: int = 100


@dataclass
class Config:
    """主配置"""
    exp_name: str = "experiment"
    seed: int = 42
    device: str = "cuda"
    
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    weight_sync: WeightSyncConfig = field(default_factory=WeightSyncConfig)
