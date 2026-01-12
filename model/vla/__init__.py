"""
VLA (Vision-Language-Action) 策略模块

实现 π0, π0.5, π0.6* 系列模型
- PI0Policy: 基础 Flow Matching VLA
- PI05Policy: 带离散动作的增强版
- ValueFunction: 价值函数网络 (用于 π0.6* RECAP)

架构参考 OpenTau (https://github.com/TensorAuto/OpenTau)
论文:
- π0: https://www.physicalintelligence.company/download/pi0.pdf
- π0.5: https://www.physicalintelligence.company/download/pi05.pdf
- RECAP: https://www.physicalintelligence.company/download/recap.pdf
"""
from .configuration import (
    PI0Config,
    PI05Config,
    ValueConfig,
    RECAPConfig,
    RewardConfig,
    NormalizationMode,
)
from .pi0_policy import PI0Policy, PI0FlowMatching
from .pi05_policy import PI05Policy, PI05FlowMatching
from .value_function import (
    ValueFunction,
    ValueModel,
    SIGLIPVisionEncoder,
    compute_n_step_returns,
    compute_advantage_indicator,
)
from .paligemma_with_expert import PaliGemmaWithExpertConfig, PaliGemmaWithExpertModel

__all__ = [
    # Configs
    "PI0Config",
    "PI05Config", 
    "ValueConfig",
    "RECAPConfig",
    "RewardConfig",
    "NormalizationMode",
    # Policies
    "PI0Policy",
    "PI05Policy",
    "ValueFunction",
    # Models
    "PI0FlowMatching",
    "PI05FlowMatching",
    "ValueModel",
    "SIGLIPVisionEncoder",
    "PaliGemmaWithExpertModel",
    "PaliGemmaWithExpertConfig",
    # Utils
    "compute_n_step_returns",
    "compute_advantage_indicator",
]
