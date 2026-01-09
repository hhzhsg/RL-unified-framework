"""
VLA-RL 配置模块

提供:
- 各模块配置 dataclass
- 预设配置工厂函数
- YAML 配置加载
"""
from .config import (
    # 配置类
    Config,
    EnvConfig,
    BufferConfig,
    ModelConfig,
    AlgorithmConfig,
    StageConfig,
    TrainingConfig,
    InferenceConfig,
    WeightSyncConfig,
    DataSourceConfig,
    # 工厂函数
    make_bc_config,
    make_td3bc_config,
    make_sac_config,
    make_recap_config,
    make_hil_config,
    # 加载函数
    load_config_from_yaml,
    get_data_config,
)

__all__ = [
    # 配置类
    "Config",
    "EnvConfig",
    "BufferConfig",
    "ModelConfig",
    "AlgorithmConfig",
    "StageConfig",
    "TrainingConfig",
    "InferenceConfig",
    "WeightSyncConfig",
    "DataSourceConfig",
    # 工厂函数
    "make_bc_config",
    "make_td3bc_config",
    "make_sac_config",
    "make_recap_config",
    "make_hil_config",
    # 加载函数
    "load_config_from_yaml",
    "get_data_config",
]
