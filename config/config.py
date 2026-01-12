"""
VLA-RL 配置定义

使用 dataclass 定义类型安全的配置结构
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import yaml


@dataclass
class EnvConfig:
    """环境配置"""
    name: str = "dummy"
    state_dim: int = 16
    action_dim: int = 4
    action_space: str = "joint"
    max_episode_steps: int = 200
    obs_cameras: List[str] = field(default_factory=lambda: ["front"])
    image_size: int = 224


@dataclass
class ModelConfig:
    """模型配置"""
    policy_type: str = "mlp_gaussian"
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])


@dataclass
class AlgorithmConfig:
    """算法配置"""
    name: str = "sac"
    lr: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    auto_alpha: bool = True
    algo_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """从 algo_kwargs 或属性中获取值"""
        if key in self.algo_kwargs:
            return self.algo_kwargs[key]
        return getattr(self, key, default)


@dataclass
class StageConfig:
    """训练阶段配置"""
    name: str = "train"
    algorithm: str = "sac"
    max_steps: int = 100000
    active_models: List[str] = field(default_factory=lambda: ["policy", "q1", "q2"])
    sample_strategy: str = "rollout_only"
    sample_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """训练配置"""
    stages: List[StageConfig] = field(default_factory=list)
    log_freq: int = 100
    save_freq: int = 10000
    checkpoint_dir: str = "./checkpoints"


@dataclass
class WeightSyncConfig:
    """权重同步配置"""
    method: str = "shared_memory"
    sync_freq: int = 100


@dataclass
class InferenceConfig:
    """推理配置"""
    device: str = "cpu"
    deterministic: bool = False
    action_horizon: int = 1
    execute_steps: int = 1
    history_len: int = 1


@dataclass
class DataSourceConfig:
    """数据源配置"""
    type: str = "buffer"
    demo_paths: List[str] = field(default_factory=list)
    camera_keys: List[str] = field(default_factory=list)
    load_images: bool = False
    rollout_capacity: int = 100000
    intervention_capacity: int = 10000


# 兼容别名
BufferConfig = DataSourceConfig


@dataclass
class Config:
    """完整配置"""
    exp_name: str = "experiment"
    seed: int = 42
    device: str = "cuda"
    
    env: EnvConfig = field(default_factory=EnvConfig)
    data: DataSourceConfig = field(default_factory=DataSourceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    weight_sync: WeightSyncConfig = field(default_factory=WeightSyncConfig)


def load_config_from_yaml(yaml_path: str, config_name: str) -> Config:
    """从 YAML 文件加载配置"""
    with open(yaml_path, 'r') as f:
        all_configs = yaml.safe_load(f)
    
    if 'configs' not in all_configs:
        raise ValueError(f"YAML 文件缺少 'configs' 键")
    
    if config_name not in all_configs['configs']:
        available = list(all_configs['configs'].keys())
        raise ValueError(f"配置 '{config_name}' 不存在。可用: {available}")
    
    cfg_dict = all_configs['configs'][config_name]
    return _dict_to_config(cfg_dict)


def _dict_to_config(d: Dict) -> Config:
    """将字典转换为 Config 对象"""
    config = Config()
    
    # 基本字段
    for key in ['exp_name', 'seed', 'device']:
        if key in d:
            setattr(config, key, d[key])
    
    # 嵌套配置
    if 'env' in d:
        config.env = EnvConfig(**d['env'])
    if 'data' in d:
        config.data = DataSourceConfig(**d['data'])
    if 'model' in d:
        config.model = ModelConfig(**d['model'])
    if 'algorithm' in d:
        algo_dict = d['algorithm'].copy()
        # 提取已知字段，其余放入 algo_kwargs
        known_fields = {'name', 'lr', 'batch_size', 'gamma', 'tau', 'alpha', 'auto_alpha', 'algo_kwargs'}
        algo_kwargs = algo_dict.pop('algo_kwargs', {})
        for key in list(algo_dict.keys()):
            if key not in known_fields:
                algo_kwargs[key] = algo_dict.pop(key)
        algo_dict['algo_kwargs'] = algo_kwargs
        config.algorithm = AlgorithmConfig(**algo_dict)
    if 'training' in d:
        training_dict = d['training'].copy()
        if 'stages' in training_dict:
            training_dict['stages'] = [StageConfig(**s) for s in training_dict['stages']]
        config.training = TrainingConfig(**training_dict)
    if 'weight_sync' in d:
        config.weight_sync = WeightSyncConfig(**d['weight_sync'])
    
    return config


def get_data_config(config: Config) -> DataSourceConfig:
    """获取数据配置"""
    return config.data
