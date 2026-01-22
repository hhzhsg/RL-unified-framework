"""HIL-SERL专用变换"""
from typing import Dict, Any
import numpy as np
from core.interfaces import TransformInterface


class InterventionTransform(TransformInterface):
    """标记干预数据"""
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["source"] = "intervention"
        return data


class JointVelocityTransform(TransformInterface):
    """添加关节速度"""
    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self._prev_pos = None
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "state" in data:
            if self._prev_pos is not None:
                vel = (data["state"] - self._prev_pos) / self.dt
                data["joint_velocity"] = vel
            self._prev_pos = data["state"].copy()
        return data
