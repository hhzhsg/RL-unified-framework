"""
任务奖励配置模板

每个 project 需要根据具体任务定义：
1. CLASSIFIER_CONFIG - classifier 相关配置
2. reward_func() - 复合奖励函数

调用场景：
- HILActorLoop: 在线交互时计算 reward
- record_demos.py: 采集 demo 时判断是否成功
- Learner: 直接使用 buffer 中已计算好的 reward

使用方式：
    from projects.your_task.reward_config import CLASSIFIER_CONFIG, reward_func
"""
import numpy as np
from typing import Dict, Any


# ============ Classifier 配置 ============
CLASSIFIER_CONFIG = {
    # Classifier checkpoint 路径（相对于项目根目录）
    "checkpoint": "checkpoints/classifier.pt",
    
    # 用于 classifier 的相机
    "cameras": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
    
    # Classifier 阈值（仅用于默认 reward_func）
    "threshold": 0.8,
    
    # 图像尺寸（与训练时一致）
    "image_size": 224,
}


# ============ 复合奖励函数 ============
def reward_func(obs: Dict[str, Any], classifier_prob: float) -> bool:
    """
    任务特定的复合奖励函数
    
    根据 classifier 输出和状态条件判断是否成功。
    需要根据具体任务修改此函数。
    
    Args:
        obs: 环境观测字典，包含 state, images 等
        classifier_prob: classifier 输出的成功概率 (0~1)
        
    Returns:
        是否成功 (True/False)
        
    示例（RAM Insertion）：
        return (
            classifier_prob > 0.85 and
            obs["state"][6] > 0.04  # z position > 0.04
        )
        
    示例（Object Handover）：
        return (
            classifier_prob > 0.75 and
            obs["state"][0] > 0.5   # gripper opened
        )
    """
    # 默认实现：只使用 classifier 阈值
    threshold = CLASSIFIER_CONFIG.get("threshold", 0.8)
    return classifier_prob > threshold


# ============ 可选：额外的状态条件 ============
def check_state_conditions(obs: Dict[str, Any]) -> bool:
    """
    检查状态条件（可选）
    
    如果任务有额外的状态条件（如位置、夹爪状态），在这里定义。
    然后在 reward_func 中调用此函数。
    
    Args:
        obs: 环境观测
        
    Returns:
        状态条件是否满足
    """
    # 默认：无额外条件
    return True


# ============ 可选：自定义 reward shaping ============
def compute_shaped_reward(
    obs: Dict[str, Any],
    next_obs: Dict[str, Any],
    classifier_prob: float,
    is_intervention: bool,
) -> float:
    """
    计算 shaped reward（可选）
    
    如果需要除了 0/1 sparse reward 之外的奖励塑形，可以在这里定义。
    HIL-SERL 默认使用 sparse reward（成功=1，失败=0）。
    
    Args:
        obs: 当前观测
        next_obs: 下一观测
        classifier_prob: classifier 成功概率
        is_intervention: 是否为人工干预
        
    Returns:
        shaped reward value
    """
    # 默认：sparse reward
    if reward_func(next_obs, classifier_prob):
        return 1.0
    return 0.0
