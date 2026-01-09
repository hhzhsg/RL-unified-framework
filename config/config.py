"""
VLA-RL 配置定义

配置层次：
- Config: 顶层配置，聚合所有子配置
- XxxConfig: 各模块配置 (dataclass)
- algo_kwargs: 算法特有参数 (Dict)
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class EnvConfig:
    """环境配置"""
    name: str = "dummy"
    obs_cameras: List[str] = field(default_factory=lambda: ["front"])
    image_size: int = 224
    state_dim: int = 14              # 机器人状态维度
    action_dim: int = 7              # 动作维度
    action_space: str = "joint"      # "joint" | "cartesian" | "delta"
    control_freq: int = 50           # 控制频率 Hz
    max_episode_steps: int = 500


@dataclass
class BufferConfig:
    """Buffer 配置"""
    max_size: int = 100000
    demo_path: Optional[str] = None  # 预加载 demo 数据路径


@dataclass
class ModelConfig:
    """模型配置"""
    policy_type: str = "mlp"         # "mlp" | "mlp_gaussian" | "vla" | "diffusion"
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    use_robot_state: bool = True     # 是否使用 robot_state 作为输入


@dataclass
class AlgorithmConfig:
    """
    算法配置
    
    通用参数直接作为字段，算法特有参数放 algo_kwargs
    """
    name: str = "bc"                 # "bc" | "sac" | "td3_bc" | "cql" | "iql"
    lr: float = 1e-4
    batch_size: int = 256
    
    # RL 通用参数
    gamma: float = 0.99              # 折扣因子
    tau: float = 0.005               # 软更新系数
    
    # SAC 参数
    alpha: float = 0.2               # 温度系数
    auto_alpha: bool = True          # 是否自动调节温度
    
    # 算法特有参数 (透传给具体算法)
    # TD3+BC: bc_alpha, policy_noise, noise_clip, policy_freq
    # CQL: cql_alpha, num_random_actions
    # IQL: expectile, beta
    algo_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default=None):
        """获取参数，优先从 algo_kwargs 获取"""
        if key in self.algo_kwargs:
            return self.algo_kwargs[key]
        if hasattr(self, key):
            return getattr(self, key)
        return default


@dataclass
class StageConfig:
    """训练阶段配置"""
    name: str = "train"
    algorithm: str = "bc"            # 该阶段使用的算法
    max_steps: int = 100000
    active_models: List[str] = field(default_factory=lambda: ["policy"])
    sample_strategy: str = "demo_only"
    sample_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """训练配置"""
    stages: List[StageConfig] = field(default_factory=lambda: [StageConfig()])
    log_freq: int = 100
    eval_freq: int = 5000
    save_freq: int = 10000
    checkpoint_dir: str = "./checkpoints"


@dataclass
class InferenceConfig:
    """推理配置"""
    device: str = "cpu"
    action_horizon: int = 1          # 预测多少步
    execute_steps: int = 1           # 执行多少步后重新预测
    history_len: int = 1             # 历史窗口长度 (1 表示不用历史)
    deterministic: bool = True


@dataclass
class WeightSyncConfig:
    """权重同步配置"""
    method: str = "queue"            # "queue" | "redis" | "shared_memory"
    sync_freq: int = 100             # 多少步同步一次


@dataclass
class DataSourceConfig:
    """数据源配置"""
    type: str = "hdf5"                           # "hdf5" | "buffer"
    demo_paths: List[str] = field(default_factory=list)
    camera_keys: List[str] = field(default_factory=lambda: ["cam_high", "cam_left_wrist", "cam_right_wrist"])
    load_images: bool = True
    rollout_capacity: int = 100000
    intervention_capacity: int = 50000


@dataclass
class Config:
    """
    主配置
    
    聚合所有子配置，作为统一入口
    """
    env: EnvConfig = field(default_factory=EnvConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    weight_sync: WeightSyncConfig = field(default_factory=WeightSyncConfig)
    
    seed: int = 42
    device: str = "cuda"
    exp_name: str = "default"


# ============ 预设配置工厂 ============

def make_bc_config() -> Config:
    """离线 BC 配置"""
    return Config(
        algorithm=AlgorithmConfig(name="bc", lr=1e-4),
        training=TrainingConfig(
            stages=[StageConfig(
                name="bc",
                algorithm="bc",
                max_steps=100000,
                sample_strategy="demo_only",
            )]
        )
    )


def make_td3bc_config() -> Config:
    """离线 TD3+BC 配置"""
    return Config(
        algorithm=AlgorithmConfig(
            name="td3_bc",
            lr=3e-4,
            algo_kwargs={
                "bc_alpha": 2.5,
                "policy_noise": 0.2,
                "noise_clip": 0.5,
                "policy_freq": 2,
            }
        ),
        training=TrainingConfig(
            stages=[StageConfig(
                name="td3bc",
                algorithm="td3_bc",
                max_steps=100000,
                active_models=["policy", "q1", "q2"],
                sample_strategy="demo_only",
            )]
        )
    )


def make_sac_config() -> Config:
    """在线 SAC 配置"""
    return Config(
        model=ModelConfig(policy_type="mlp_gaussian"),
        algorithm=AlgorithmConfig(name="sac", lr=3e-4, auto_alpha=True),
        training=TrainingConfig(
            stages=[StageConfig(
                name="sac",
                algorithm="sac",
                max_steps=1000000,
                active_models=["policy", "q1", "q2"],
                sample_strategy="rollout_only",
            )]
        ),
        weight_sync=WeightSyncConfig(sync_freq=100),
    )


def make_recap_config() -> Config:
    """RECAP (π₀*) 配置: 两阶段训练"""
    return Config(
        training=TrainingConfig(
            stages=[
                StageConfig(
                    name="train_vf",
                    algorithm="vf_regression",
                    max_steps=50000,
                    active_models=["vf"],
                    sample_strategy="demo_only",
                ),
                StageConfig(
                    name="train_policy",
                    algorithm="awr",
                    max_steps=50000,
                    active_models=["policy"],
                    sample_strategy="demo_only",
                ),
            ]
        )
    )


def make_hil_config() -> Config:
    """Human-in-the-Loop 配置"""
    return Config(
        model=ModelConfig(policy_type="mlp_gaussian"),
        algorithm=AlgorithmConfig(name="sac", lr=3e-4),
        training=TrainingConfig(
            stages=[StageConfig(
                name="hil",
                algorithm="sac",
                max_steps=500000,
                active_models=["policy", "q1", "q2"],
                sample_strategy="mixed",
                sample_kwargs={"demo_ratio": 0.5, "intervention_ratio": 0.2},
            )]
        )
    )


# ============ YAML 配置加载 ============

def _load_yaml_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    import yaml
    from pathlib import Path
    
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _dict_to_dataclass(data: dict, cls):
    """将字典转换为 dataclass，只使用 dataclass 中定义的字段"""
    import dataclasses
    if not dataclasses.is_dataclass(cls):
        return data
    
    field_names = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered)


def load_config_from_yaml(config_path: str, config_name: str = "offline_bc") -> Config:
    """
    从 YAML 文件加载配置
    
    Args:
        config_path: YAML 配置文件路径
        config_name: 配置名称 (YAML 中 configs 下的 key)
    
    Returns:
        Config 对象
    """
    yaml_data = _load_yaml_config(config_path)
    
    if "configs" not in yaml_data:
        raise ValueError("YAML 文件必须包含 'configs' 字段")
    
    if config_name not in yaml_data["configs"]:
        available = list(yaml_data["configs"].keys())
        raise ValueError(f"配置 '{config_name}' 不存在。可用: {available}")
    
    cfg = yaml_data["configs"][config_name]
    
    # 解析各子配置
    env_config = _dict_to_dataclass(cfg.get("env", {}), EnvConfig)
    buffer_config = _dict_to_dataclass(cfg.get("buffer", {}), BufferConfig)
    model_config = _dict_to_dataclass(cfg.get("model", {}), ModelConfig)
    inference_config = _dict_to_dataclass(cfg.get("inference", {}), InferenceConfig)
    weight_sync_config = _dict_to_dataclass(cfg.get("weight_sync", {}), WeightSyncConfig)
    
    # 解析 algorithm 配置 (支持 algo_kwargs)
    algo_dict = cfg.get("algorithm", {})
    algo_kwargs = algo_dict.pop("algo_kwargs", {})
    # 将 YAML 中的额外字段也放入 algo_kwargs
    known_fields = {f.name for f in __import__('dataclasses').fields(AlgorithmConfig)}
    for key in list(algo_dict.keys()):
        if key not in known_fields and key != "algo_kwargs":
            algo_kwargs[key] = algo_dict.pop(key)
    algo_dict["algo_kwargs"] = algo_kwargs
    algo_config = _dict_to_dataclass(algo_dict, AlgorithmConfig)
    
    # 解析 training 配置 (包含 stages)
    training_dict = cfg.get("training", {})
    stages = []
    for stage_dict in training_dict.get("stages", []):
        stages.append(_dict_to_dataclass(stage_dict, StageConfig))
    
    training_config = TrainingConfig(
        stages=stages if stages else [StageConfig()],
        log_freq=training_dict.get("log_freq", 100),
        eval_freq=training_dict.get("eval_freq", 5000),
        save_freq=training_dict.get("save_freq", 10000),
        checkpoint_dir=training_dict.get("checkpoint_dir", "./checkpoints"),
    )
    
    # 解析数据配置
    data_config = _dict_to_dataclass(cfg.get("data", {}), DataSourceConfig)
    
    # 构建主配置
    config = Config(
        env=env_config,
        buffer=buffer_config,
        model=model_config,
        algorithm=algo_config,
        training=training_config,
        inference=inference_config,
        weight_sync=weight_sync_config,
        seed=cfg.get("seed", 42),
        device=cfg.get("device", "cuda"),
        exp_name=cfg.get("exp_name", config_name),
    )
    
    # 将 data_config 作为额外属性附加
    config._data_config = data_config
    
    return config


def get_data_config(config: Config) -> DataSourceConfig:
    """获取 Config 对象中的数据配置"""
    return getattr(config, '_data_config', DataSourceConfig())
