"""
VLA 策略配置类

定义 π0, π0.5, Value Function 的配置参数
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Literal, Optional, Any
from enum import Enum


class NormalizationMode(str, Enum):
    """归一化模式"""
    IDENTITY = "identity"
    MEAN_STD = "mean_std"
    MIN_MAX = "min_max"


@dataclass
class RewardConfig:
    """π0.6* 奖励计算配置"""
    number_of_bins: int = 201  # Value bins 数量
    C_neg: float = -1000.0     # 失败惩罚
    reward_normalizer: int = 400  # 最大 episode 长度
    N_steps_look_ahead: int = 50  # n-step return 步数


@dataclass
class PI0Config:
    """
    π0 Policy 配置
    
    π0: Vision-Language-Action Flow Model
    Paper: https://www.physicalintelligence.company/download/pi0.pdf
    
    架构:
    - PaliGemma (3B) 作为 VLM backbone
    - Gemma Expert 作为 action decoder
    - Flow Matching 生成连续动作
    """
    # 输入/输出结构
    n_obs_steps: int = 1          # 观测历史长度
    chunk_size: int = 50          # action chunk 大小
    n_action_steps: int = 50      # 预测动作步数
    
    # 状态/动作维度 (较短的会 padding)
    max_state_dim: int = 32
    max_action_dim: int = 32
    
    # 图像预处理
    resize_imgs_with_padding: Tuple[int, int] = (224, 224)
    empty_cameras: int = 0  # 空相机数量 (用于某些 sim)
    
    # 归一化配置
    normalization_mapping: Dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )
    
    # Tokenizer
    tokenizer_max_length: int = 48
    
    # 模型结构
    proj_width: int = 1024  # projection 层宽度
    dropout: float = 0.1
    
    # Flow Matching 解码
    num_steps: int = 10  # denoising steps
    
    # AWR (Advantage-Weighted Regression) 用于 π0.6*
    advantage_threshold: float = 0.0
    advantage: Literal["ignore", "on", "use"] = "use"
    # "use": 使用数据集中的 advantage 值
    # "ignore": 禁用 advantage conditioning
    # "on": 始终设为 True (仅用于 expert demo)
    
    # 初始化策略
    init_strategy: Literal["no_init", "full_he_init", "expert_only_he_init"] = "full_he_init"
    
    # 注意力实现
    use_cache: bool = True
    attention_implementation: str = "eager"  # "eager" or "fa2" (flash attention)
    
    # 微调设置
    freeze_vision_encoder: bool = True
    train_expert_only: bool = False
    train_state_proj: bool = True
    
    # 优化器预设
    optimizer_lr: float = 2.5e-5
    optimizer_betas: Tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    
    # Scheduler 预设
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6
    
    # 图像特征名
    image_features: List[str] = field(
        default_factory=lambda: ["observation.image"]
    )
    
    # 设备
    device: str = "cuda"


@dataclass
class PI05Config(PI0Config):
    """
    π0.5 Policy 配置
    
    π0.5 在 π0 基础上增加:
    - 离散动作 tokenization (FAST tokenizer)
    - Knowledge Insulation (KI) - VLM 和 Expert 之间的梯度隔离
    - Adaptive RMS Norm (AdaRMS)
    
    Paper: https://www.physicalintelligence.company/download/pi05.pdf
    """
    # Tokenizer 配置
    tokenizer_max_length: int = 256
    
    # 离散动作配置
    discrete_action_max_length: int = 32
    discrete_action_vocab_size: Optional[int] = None  # 从 processor 自动获取
    
    # π0.5 特有配置
    use_knowledge_insulation: bool = True  # 是否启用 KI
    use_adarms: bool = True  # 是否使用 AdaRMS


@dataclass  
class ValueConfig:
    """
    Value Function 配置
    
    用于 π0.6* RECAP (Reinforcement Learning with Expert Advantage Conditioning)
    
    架构:
    - SIGLIP 视觉编码器
    - Gemma 3 (270M) 语言模型
    - Value Head 输出离散化 bins
    """
    # 模型架构
    image_size: int = 224
    vision_dim: int = 1152  # SIGLIP hidden dim
    vision_depth: int = 27  # SIGLIP layers
    hidden_dim: int = 1024  # Gemma 3 hidden dim
    num_layers: int = 26  # Gemma 3 layers
    num_heads: int = 8
    ff_dim: int = 4096
    vocab_size: int = 256000
    dropout: float = 0.0
    
    # Value bins
    number_of_bins: int = 201
    
    # 输入结构
    n_obs_steps: int = 1
    chunk_size: int = 50
    
    # 状态维度
    max_state_dim: int = 32
    
    # 归一化
    normalization_mapping: Dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "VALUE": NormalizationMode.MEAN_STD,
        }
    )
    
    # 图像预处理
    resize_imgs_with_padding: Tuple[int, int] = (224, 224)
    empty_cameras: int = 0
    
    # Tokenizer
    tokenizer_max_length: int = 48
    
    # 奖励配置 (用于 bin 计算)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    
    # 优化器预设
    optimizer_lr: float = 2.5e-5
    optimizer_betas: Tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    
    # Scheduler 预设
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6
    
    # 图像特征名
    image_features: List[str] = field(
        default_factory=lambda: ["observation.image"]
    )
    
    # 设备
    device: str = "cuda"


@dataclass
class RECAPConfig:
    """
    RECAP (π0.6*) 训练配置
    
    三阶段训练流程:
    1. SFT: 在 demo 数据上预训练 VLA
    2. Value Training: 训练价值函数
    3. Offline RL: AWR 微调 VLA
    """
    # 子配置
    policy_config: Optional[Any] = None  # PI0Config 或 PI05Config
    value_config: Optional[Any] = None  # ValueConfig
    
    # Stage 1: SFT 配置
    sft_steps: int = 100_000
    sft_batch_size: int = 32
    
    # Stage 2: Value Function 配置
    value_steps: int = 80_000
    value_batch_size: int = 32
    
    # Stage 3: Offline RL (AWR) 配置
    awr_steps: int = 50_000
    rl_iterations: int = 3  # RL iteration 次数
    episodes_per_iteration: int = 300  # 每次迭代收集的 episode 数
    rl_steps_per_iteration: int = 10_000
    
    # 通用训练配置
    batch_size: int = 32
    policy_lr: float = 1e-4
    value_lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    
    # Advantage 配置
    advantage_threshold: float = 0.0  # A(s) > threshold 才训练
    advantage_percentile: float = 0.3  # 正 advantage 的比例阈值
    
    # 奖励配置
    reward_config: RewardConfig = field(default_factory=RewardConfig)
