"""
配置加载器

从 YAML 文件加载配置
"""
from typing import Dict, Any
import yaml

from .base import (
    Config, EnvConfig, ModelConfig, AlgorithmConfig,
    DataConfig, TrainingConfig, InferenceConfig, WeightSyncConfig,
)


def load_config_from_yaml(yaml_path: str, config_name: str) -> Config:
    """
    从 YAML 文件加载配置
    
    Args:
        yaml_path: YAML 文件路径
        config_name: 配置名称
        
    Returns:
        配置对象
    """
    with open(yaml_path, 'r') as f:
        all_configs = yaml.safe_load(f)
    
    if 'configs' not in all_configs:
        raise ValueError("YAML file missing 'configs' key")
    
    if config_name not in all_configs['configs']:
        available = list(all_configs['configs'].keys())
        raise ValueError(f"Config '{config_name}' not found. Available: {available}")
    
    cfg_dict = all_configs['configs'][config_name]
    return _dict_to_config(cfg_dict)


def _dict_to_config(d: Dict[str, Any]) -> Config:
    """将字典转换为 Config 对象"""
    config = Config()
    
    # 基本字段
    for key in ['exp_name', 'seed', 'device']:
        if key in d:
            setattr(config, key, d[key])
    
    # 嵌套配置
    if 'env' in d:
        config.env = EnvConfig(**d['env'])
    
    if 'model' in d:
        config.model = ModelConfig(**d['model'])
    
    if 'algorithm' in d:
        algo_dict = d['algorithm'].copy()
        # 提取已知字段，其余放入 algo_kwargs
        known_fields = {'name', 'lr', 'batch_size', 'gamma', 'tau', 'alpha', 'auto_alpha', 'grad_clip', 'algo_kwargs'}
        algo_kwargs = algo_dict.pop('algo_kwargs', {})
        for key in list(algo_dict.keys()):
            if key not in known_fields:
                algo_kwargs[key] = algo_dict.pop(key)
        algo_dict['algo_kwargs'] = algo_kwargs
        config.algorithm = AlgorithmConfig(**algo_dict)
    
    if 'data' in d:
        config.data = DataConfig(**d['data'])
    
    if 'training' in d:
        config.training = TrainingConfig(**d['training'])
    
    if 'inference' in d:
        config.inference = InferenceConfig(**d['inference'])
    
    if 'weight_sync' in d:
        config.weight_sync = WeightSyncConfig(**d['weight_sync'])
    
    return config


def save_config_to_yaml(config: Config, yaml_path: str, config_name: str = "default"):
    """
    保存配置到 YAML 文件
    
    Args:
        config: 配置对象
        yaml_path: 保存路径
        config_name: 配置名称
    """
    from dataclasses import asdict
    
    cfg_dict = asdict(config)
    
    with open(yaml_path, 'w') as f:
        yaml.dump({'configs': {config_name: cfg_dict}}, f, default_flow_style=False)
