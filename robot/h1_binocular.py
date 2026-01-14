"""
H1 双目机器人适配器
"""
from typing import Dict, Any, List
import numpy as np

from .base import BaseRobotAdapter, RobotSpec


# 双目机器人规格
BINOCULAR_SPEC = RobotSpec(
    name="binocular_single_arm",
    state_dim=15,
    action_dim=15,
    camera_keys=[
        "base_0_rgb_0",      # 头部左相机
        "base_0_rgb_1",      # 头部右相机
        "right_wrist_0_rgb_0",  # 手腕左相机
        "right_wrist_0_rgb_1",  # 手腕右相机
    ],
    state_keys=[
        "arm_position",      # 7 维
        "effector_position", # 1 维
        "waist_position",    # 2 维
        "head_position",     # 2 维
        "base_velocity",     # 3 维
    ],
    action_keys=[
        "arm_position",
        "effector_position",
        "waist_position",
        "head_position",
        "base_velocity",
    ],
)


class BinocularAdapter(BaseRobotAdapter):
    """
    双目机器人适配器
    
    相机配置:
    - 头部: base_0_rgb_0 (左), base_0_rgb_1 (右)
    - 手腕: right_wrist_0_rgb_0 (左), right_wrist_0_rgb_1 (右)
    
    状态维度: 15
    - arm: 7 (从位置索引 7 开始取)
    - effector: 1 (最后一个)
    - waist: 2
    - head: 2
    - base: 3
    
    动作维度: 15 (同状态)
    """
    
    # HDF5 路径映射
    IMAGE_MAPPING = {
        "base_0_rgb_0": "/observation/images/v4l2/cam_high_0/color",
        "base_0_rgb_1": "/observation/images/v4l2/cam_high_1/color",
        "right_wrist_0_rgb_0": "/observation/images/v4l2/cam_right_wrist_0/color",
        "right_wrist_0_rgb_1": "/observation/images/v4l2/cam_right_wrist_1/color",
    }
    
    STATE_MAPPING = {
        "arm_position": ("/observation/state/arm/position", slice(7, None)),
        "effector_position": ("/observation/state/effector/position", slice(-1, None)),
        "waist_position": ("/observation/state/waist/position", slice(None)),
        "head_position": ("/observation/state/head/position", slice(None)),
        "base_velocity": ("/observation/state/base/velocity", slice(None)),
    }
    
    ACTION_MAPPING = {
        "arm_position": "/action/arm/position",
        "effector_position": "/action/effector/position",
        "waist_position": "/action/waist/position",
        "head_position": "/action/head/position",
        "base_velocity": "/action/base/velocity",
    }
    
    def __init__(self):
        super().__init__(BINOCULAR_SPEC)
    
    def preprocess(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        预处理原始 HDF5 数据
        
        Args:
            raw_data: {hdf5_path: value} 格式的原始数据
            
        Returns:
            标准化格式:
            {
                "images": {camera_name: image},
                "state": np.ndarray,
                "action": np.ndarray,
            }
        """
        result = {
            "images": {},
            "state": None,
            "action": None,
        }
        
        # 处理图像
        for cam_name, hdf5_path in self.IMAGE_MAPPING.items():
            if hdf5_path in raw_data:
                result["images"][cam_name] = raw_data[hdf5_path]
        
        # 处理状态
        state_parts = []
        for key, (hdf5_path, slc) in self.STATE_MAPPING.items():
            if hdf5_path in raw_data:
                data = raw_data[hdf5_path]
                if slc:
                    data = data[slc]
                state_parts.append(np.array(data).flatten())
        
        if state_parts:
            result["state"] = np.concatenate(state_parts).astype(np.float32)
        
        # 处理动作
        action_parts = []
        for key, hdf5_path in self.ACTION_MAPPING.items():
            if hdf5_path in raw_data:
                action_parts.append(np.array(raw_data[hdf5_path]).flatten())
        
        if action_parts:
            result["action"] = np.concatenate(action_parts).astype(np.float32)
        
        return result
    
    def postprocess(self, model_output: np.ndarray) -> np.ndarray:
        """
        后处理模型输出
        
        直接返回前 15 维作为动作
        """
        return model_output[:self.action_dim].astype(np.float32)
    
    def get_state_mapping(self) -> Dict[str, str]:
        return {k: v[0] for k, v in self.STATE_MAPPING.items()}
    
    def get_action_mapping(self) -> Dict[str, str]:
        return self.ACTION_MAPPING.copy()
