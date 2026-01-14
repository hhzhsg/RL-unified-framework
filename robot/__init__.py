"""
机器人适配器模块

支持不同机器人平台的数据适配:
- BinocularAdapter: 双目单臂机器人
"""
from .base import BaseRobotAdapter, RobotSpec
from .h1_binocular import BinocularAdapter, BINOCULAR_SPEC

# 适配器注册表
ROBOT_REGISTRY = {
    "binocular": BinocularAdapter,
    "binocular_single_arm": BinocularAdapter,
}


def create_robot_adapter(name: str) -> BaseRobotAdapter:
    """
    创建机器人适配器
    
    Args:
        name: 机器人类型
        
    Returns:
        适配器实例
    """
    if name not in ROBOT_REGISTRY:
        raise ValueError(f"Unknown robot: {name}. Available: {list(ROBOT_REGISTRY.keys())}")
    return ROBOT_REGISTRY[name]()


__all__ = [
    "BaseRobotAdapter",
    "RobotSpec",
    "BinocularAdapter",
    "BINOCULAR_SPEC",
    "ROBOT_REGISTRY",
    "create_robot_adapter",
]
