"""
VR Controller Wrapper

通过 VR 控制器进行人类干预
"""
from typing import Any, Dict, Optional, Tuple
import numpy as np

from .base_intervention import BaseInterventionWrapper


class VRWrapper(BaseInterventionWrapper):
    """
    VR 控制器干预 Wrapper
    
    支持：
    - Meta Quest / HTC Vive 等 VR 控制器
    - 通过按钮激活/释放干预
    - 6DOF 姿态映射到机器人动作
    
    使用示例：
        from env.wrappers import VRWrapper
        
        env = VRWrapper(
            base_env,
            vr_device="quest",
            action_scale=1.0,
            trigger_button="grip",
        )
        
        # Actor loop 中正常调用
        action = policy.act(obs)
        obs, reward, done, truncated, info = env.step(action)
    """
    
    def __init__(
        self,
        env,
        vr_device: str = "quest",
        action_scale: float = 1.0,
        trigger_button: str = "grip",  # "grip" | "trigger" | "any"
        position_dims: Tuple[int, ...] = (0, 1, 2),  # xyz 映射到动作的哪些维度
        rotation_dims: Tuple[int, ...] = (3, 4, 5),  # rpy 映射到动作的哪些维度
        gripper_dim: Optional[int] = 6,  # 夹爪映射到哪个维度
        intervention_threshold: float = 0.001,
        sticky_duration: float = 0.3,
    ):
        """
        Args:
            env: 被包装的环境
            vr_device: VR 设备类型 ("quest" | "vive" | "index")
            action_scale: 动作缩放因子
            trigger_button: 触发干预的按钮
            position_dims: 位置映射到动作空间的维度
            rotation_dims: 旋转映射到动作空间的维度
            gripper_dim: 夹爪映射到动作空间的维度
            intervention_threshold: 干预激活阈值
            sticky_duration: 干预粘滞时间
        """
        super().__init__(env, intervention_threshold, sticky_duration)
        
        self.vr_device = vr_device
        self.action_scale = action_scale
        self.trigger_button = trigger_button
        self.position_dims = position_dims
        self.rotation_dims = rotation_dims
        self.gripper_dim = gripper_dim
        
        # VR 设备连接
        self._vr_client: Optional[Any] = None
        self._connected = False
        self._button_pressed = False
        
        # 动作空间维度
        self._action_dim = env.action_space.shape[0]
        
        # 尝试连接 VR 设备
        self._connect_vr()
    
    def _connect_vr(self) -> bool:
        """连接 VR 设备"""
        try:
            # 这里根据实际 VR SDK 实现
            # 示例：使用假设的 VR 客户端
            if self.vr_device == "quest":
                self._vr_client = self._create_quest_client()
            elif self.vr_device == "vive":
                self._vr_client = self._create_vive_client()
            else:
                print(f"[VRWrapper] Unknown VR device: {self.vr_device}")
                return False
            
            self._connected = True
            print(f"[VRWrapper] Connected to {self.vr_device}")
            return True
        except Exception as e:
            print(f"[VRWrapper] Failed to connect: {e}")
            self._connected = False
            return False
    
    def _create_quest_client(self) -> Any:
        """创建 Meta Quest 客户端（示例实现）"""
        # TODO: 替换为实际的 Quest SDK 调用
        # 例如使用 OpenXR 或 Oculus SDK
        return MockVRClient()
    
    def _create_vive_client(self) -> Any:
        """创建 HTC Vive 客户端（示例实现）"""
        # TODO: 替换为实际的 OpenVR/SteamVR SDK 调用
        return MockVRClient()
    
    def get_intervention_action(self) -> Optional[np.ndarray]:
        """
        从 VR 控制器读取动作
        
        Returns:
            动作数组，如果没有输入则返回 None
        """
        if not self._connected or self._vr_client is None:
            return None
        
        try:
            # 读取控制器状态
            state = self._vr_client.get_controller_state()
            if state is None:
                return None
            
            # 检查按钮是否按下
            self._button_pressed = self._check_button_pressed(state)
            if not self._button_pressed:
                return None
            
            # 构建动作
            action = np.zeros(self._action_dim, dtype=np.float32)
            
            # 位置 (delta position)
            if state.get("position") is not None:
                pos_delta = np.array(state["position"]) * self.action_scale
                for i, dim in enumerate(self.position_dims):
                    if dim < self._action_dim and i < len(pos_delta):
                        action[dim] = pos_delta[i]
            
            # 旋转 (delta rotation)
            if state.get("rotation") is not None:
                rot_delta = np.array(state["rotation"]) * self.action_scale
                for i, dim in enumerate(self.rotation_dims):
                    if dim < self._action_dim and i < len(rot_delta):
                        action[dim] = rot_delta[i]
            
            # 夹爪
            if self.gripper_dim is not None and self.gripper_dim < self._action_dim:
                if state.get("gripper") is not None:
                    action[self.gripper_dim] = state["gripper"]
            
            return action
            
        except Exception as e:
            print(f"[VRWrapper] Error reading VR state: {e}")
            return None
    
    def _check_button_pressed(self, state: Dict[str, Any]) -> bool:
        """检查触发按钮是否按下"""
        buttons = state.get("buttons", {})
        
        if self.trigger_button == "grip":
            return buttons.get("grip", False)
        elif self.trigger_button == "trigger":
            return buttons.get("trigger", False)
        elif self.trigger_button == "any":
            return any(buttons.values())
        else:
            return buttons.get(self.trigger_button, False)
    
    def is_intervention_active(self, action: Optional[np.ndarray]) -> bool:
        """
        判断干预是否激活
        
        对于 VR：只有按钮按下时才激活
        """
        if action is None:
            return False
        
        # 按钮按下且有实际动作
        return self._button_pressed and np.linalg.norm(action) > self.intervention_threshold
    
    def close(self):
        """关闭 VR 连接"""
        if self._vr_client is not None:
            try:
                self._vr_client.close()
            except:
                pass
        self._connected = False
        super().close() if hasattr(super(), 'close') else None


# ============ Mock VR Client（用于测试） ============

class MockVRClient:
    """
    模拟 VR 客户端，用于测试
    
    实际使用时替换为真实的 VR SDK 客户端
    """
    
    def __init__(self):
        self._active = False
        self._position = np.zeros(3)
        self._rotation = np.zeros(3)
    
    def get_controller_state(self) -> Optional[Dict[str, Any]]:
        """获取控制器状态"""
        # 模拟：返回空状态（无干预）
        return {
            "position": self._position.tolist(),
            "rotation": self._rotation.tolist(),
            "gripper": 0.0,
            "buttons": {
                "grip": self._active,
                "trigger": False,
            }
        }
    
    def set_active(self, active: bool):
        """设置是否激活（用于测试）"""
        self._active = active
    
    def set_position(self, position: np.ndarray):
        """设置位置（用于测试）"""
        self._position = position
    
    def close(self):
        """关闭连接"""
        pass
