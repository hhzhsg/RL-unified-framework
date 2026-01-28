      
"""
H1机器人环境（纯 ZCM 订阅模式）

通过 ZCM 订阅 topic 获取机器人观测，通过 policy_action 发布动作。
参考 Zerith_H1_Deploy/utils/real_env.py 实现。

观测来源（ZCM 订阅）：
  - upper_joint_state: 14 关节位置/速度/力矩
  - gripper_state: 2 夹爪开合度
  - waist_state: 腰部状态
  - left_arm_actual / right_arm_actual: 末端执行器位姿
  - button_state: VR 按钮状态
  - chassis_control: 底盘速度
  - head_euler: 头部状态

动作发布（ZCM）：
  - policy_action: PolicyAction 消息，包含关节动作

干预检测（ZCM 订阅）：
  - policy_state: PolicyState 消息，operator_type 字段
"""
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import collections
import time
import os
import sys
import torch

from ..base_env import BaseEnv
from core.orchestration import register_env

from zerocm import ZCM
from idl.python.PolicyAction import PolicyAction
from idl.python.PolicyState import PolicyState


@register_env("h1_robot_act")
class H1ActEnv(BaseEnv):
    """H1机器人环境（纯 ZCM 订阅模式）"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # ========== 观测状态缓存 ==========
        # 观测维度规范: 37 = arm(14) + arm_vel(14) + gripper(2) + waist(3) + head(2) + base(2)
        # 动作维度规范: 23 = arm(14) + gripper(2) + waist(3) + head(2) + base(2)
        
        # 关节状态 (UpperJointState)
        self.joint_state = np.zeros(14)    # left arm 7 + right arm 7
        self.joint_vel = np.zeros(14)
        self.joint_torque = np.zeros(14)
        
        # 夹爪状态 (GripperState)
        self.gripper_state = np.zeros(2)   # left + right
        self.gripper_vel = np.zeros(2)
        
        # 腰部状态 (WaistState) - 3 维: height, pitch, yaw
        self.waist_state = np.zeros(3)
        self.waist_vel = np.zeros(3)
        
        # 头部状态 (HeadState) - 2 维: pitch, yaw
        self.head_state = np.zeros(2)
        self.head_vel = np.zeros(2)
        
        # 底盘状态 (ChassisState) - 2 维: x_vel, yaw_vel
        self.chassis_state = np.zeros(2)
        
        # 末端执行器状态 (可选)
        self.left_eef_state = np.zeros(6)  # xyz + rpy
        self.right_eef_state = np.zeros(6)
        
        # 控制目标
        self.left_action = np.zeros(6)
        self.right_action = np.zeros(6)
        self.gripper_control = np.zeros(2)
        self.chassis_control = np.zeros(2)
        self.buttons = np.zeros(4)
        self.camera_state = np.zeros(6)
        self.head_target = np.zeros(6)
        self.reset_state = 0
        
        # ========== 坐标变换矩阵（与 real_env.py 一致）==========
        self.T_l_init2base = np.array([
            [ 0.98883089,  0.13437272, -0.06447828,  0.46326353],
            [-0.1451586 ,  0.96638998, -0.21217774,  0.1982163 ],
            [ 0.03380026,  0.21916747,  0.97510162,  0.19514862],
            [ 0.        ,  0.        ,  0.        ,  1.        ],
        ])
        self.T_r_init2base = np.array([
            [0.98883089, -0.13437272, -0.06447828,  0.46338985],
            [0.1451586 ,  0.96638998,  0.21217774, -0.19829796],
            [0.03380026, -0.21916747,  0.97510162,  0.19535466],
            [0.        ,  0.        ,  0.        ,  1.        ],
        ])
        
        # ========== 相机（SDK 直接调用）==========
        # 相机通过独立的 gRPC 服务获取，端口 50051
        self.camera_names = config.get("camera_names", ['v4l2/cam_high', 'v4l2/cam_right_wrist'])
        self.use_camera = config.get("use_camera", False)  # 默认禁用
        self.split_stereo = config.get("split_stereo", True)  # 双目分割
        self.camera_grpc_target = config.get("camera_grpc_target", "localhost:50051")
        self.image_recorder = None
        
        if self.use_camera:
            try:
                from .camera_sdk import ImageRecorder
                self.image_recorder = ImageRecorder(
                    camera_names=self.camera_names,
                    grpc_target=self.camera_grpc_target,
                    split_stereo=self.split_stereo,
                )
                print(f"[H1RobotEnv] 相机初始化成功: {self.camera_names}")
            except Exception as e:
                print(f"[H1RobotEnv] 相机初始化失败: {e}")
                self.image_recorder = None
        
        # ========== 调试选项 ==========
        self.dry_run = config.get("dry_run", False)  # 只读观测，不下发动作
        if self.dry_run:
            print("[H1RobotEnv] ⚠️ DRY-RUN 模式：不下发动作")
        
        # ========== HIL 干预状态 ==========
        self.operator_type = "policy"
        self.done = False
        
        # ========== ZCM 通信 ==========
        # ipcshm: 本地共享内存（同一机器）
        # udpm://239.255.76.67:7667: UDP 多播（跨机器，需要同一局域网）
        default_zcm_url = os.environ.get("H1_ZCM_URL", "ipcshm")
        zcm_url = config.get("zcm_url", default_zcm_url)
        self.zcm_node = ZCM(zcm_url)
        
        # 导入旧版 IDL 消息类型
        self._setup_idl_subscriptions(config)
        
        # 订阅 PolicyState（干预检测）
        self.zcm_node.subscribe("policy_state", PolicyState, self._policy_state_handler)
        
        # 启动 ZCM
        self.zcm_node.start()
        
        # PolicyAction 用于发布动作
        self.policy_action = PolicyAction()
        self.policy_state = PolicyState()
        
        print(f"[H1RobotEnv] ZCM 初始化完成, url={zcm_url}")
    
    def _setup_idl_subscriptions(self, config: Dict[str, Any]) -> None:
        """
        设置观测相关的 ZCM 订阅（使用 idl_python 格式）
        
        消息类型：UpperJointState, GripperState, WaistState 等
        """
        
        # IDL 模块路径配置
        idl_paths = config.get("idl_python_paths", [])
        env_path = os.environ.get("H1_IDL_PYTHON_PATH", "")
        if env_path:
            idl_paths.append(env_path)
        # 默认路径
        idl_paths.extend([
            "/home/robot/Teleop_whole_body_sim/teleop/scripts",    # 机器人
            os.path.dirname(os.path.abspath(__file__)),
        ])
        for p in idl_paths:
            if p and p not in sys.path:
                sys.path.insert(0, p)
        
        # 导入 IDL 消息类型
        import importlib
        idl_module = config.get("idl_module", "idl_python")
        
        try:
            UpperJointState = importlib.import_module(f"{idl_module}.UpperJointState").UpperJointState
            GripperState = importlib.import_module(f"{idl_module}.GripperState").GripperState
            WaistState = importlib.import_module(f"{idl_module}.WaistState").WaistState
            HeadState = importlib.import_module(f"{idl_module}.HeadState").HeadState
            ChassisState = importlib.import_module(f"{idl_module}.ChassisState").ChassisState
            PolicyState = importlib.import_module(f"{idl_module}.PolicyState").PolicyState
            SystemInit = importlib.import_module(f"{idl_module}.SystemInit").SystemInit
            GripperControl = importlib.import_module(f"{idl_module}.GripperControl").GripperControl
            ChassisControl = importlib.import_module(f"{idl_module}.ChassisControl").ChassisControl
            
            # 保存消息类型用于发布
            self._system_init_t = SystemInit
            self._gripper_control_t = GripperControl
            self._chassis_control_t = ChassisControl
            self._head_state_t = HeadState
            self._chassis_state_t = ChassisState
            
            print(f"[H1RobotEnv] IDL 模块加载成功: {idl_module}")
            
        except ImportError as e:
            raise ImportError(
                f"无法导入 ZCM IDL 模块 '{idl_module}'。\n"
                f"请设置 H1_IDL_PYTHON_PATH 环境变量，或通过 config['idl_python_paths'] 指定路径。\n"
                f"原始错误: {e}"
            )
        
        # 订阅观测 topic
        self.zcm_node.subscribe('upper_joint_state', UpperJointState, self._upper_joint_state_handler)
        self.zcm_node.subscribe('gripper_state', GripperState, self._gripper_state_handler)
        self.zcm_node.subscribe('waist_state', WaistState, self._waist_state_handler)
        self.zcm_node.subscribe('head_state', HeadState, self._head_state_handler)
        self.zcm_node.subscribe('chassis_state', ChassisState, self._chassis_state_handler)
        
        # 订阅 PolicyState（干预检测）
        self.zcm_node.subscribe('policy_state', PolicyState, self._policy_state_handler)
        print(f"[H1RobotEnv] 已订阅 5 个观测 topic + policy_state（干预检测）")
    
    # ========== ZCM 回调（观测）==========
    
    def _upper_joint_state_handler(self, channel, msg):
        """UpperJointState: 14 关节状态"""
        self.joint_state = np.array(msg.position_actual)
        self.joint_vel = np.array(msg.speed_actual)
        self.joint_torque = np.array(msg.torque_actual)
    
    def _gripper_state_handler(self, channel, msg):
        """GripperState: 夹爪状态 (2 维)"""
        self.gripper_state = np.array(msg.position_actual) if hasattr(msg, 'position_actual') else np.zeros(2)
        self.gripper_vel = np.array(msg.speed_actual) if hasattr(msg, 'speed_actual') else np.zeros(2)
    
    def _waist_state_handler(self, channel, msg):
        """WaistState: 腰部状态 (3 维: height, pitch, yaw)"""
        self.waist_state = np.array(msg.position_actual) if hasattr(msg, 'position_actual') else np.zeros(3)
        self.waist_vel = np.array(msg.speed_actual) if hasattr(msg, 'speed_actual') else np.zeros(3)
    
    def _head_state_handler(self, channel, msg):
        """HeadState: 头部状态 (2 维: pitch, yaw)"""
        self.head_state = np.array(msg.position_actual) if hasattr(msg, 'position_actual') else np.zeros(2)
        self.head_vel = np.array(msg.speed_actual) if hasattr(msg, 'speed_actual') else np.zeros(2)
    
    def _chassis_state_handler(self, channel, msg):
        """ChassisState: 底盘状态 (2 维: x_vel, yaw_vel)"""
        self.chassis_state = np.array(msg.speed_actual) if hasattr(msg, 'speed_actual') else np.zeros(2)
    
    def _policy_state_handler(self, channel, msg):
        """ZCM 回调：接收 PolicyState，更新 operator_type（干预检测）"""
        self.policy_state = msg
        old_type = self.operator_type
        self.operator_type = msg.operator_type
        
        # 状态变化时打印
        if old_type != self.operator_type:
            print(f"[H1] operator: {old_type} -> {self.operator_type}")
    
    # ========== Gym 接口 ==========
    
    @property
    def observation_space(self) -> Dict[str, Any]:
        """
        ACT 观测空间 (15 维 qpos + images):
        - right_arm/position: 7 (右臂关节)
        - right_gripper/position: 1 (右 gripper)
        - waist/position: 3 (height, pitch, yaw)
        - head/position: 2 (pitch, yaw)
        - base/velocity: 2 (x, yaw)
        """
        return {
            "qpos": {"shape": (15,), "dtype": "float32"},
            "images": {"shape": (4, 3, 480, 640), "dtype": "float32"},
        }
    
    @property
    def action_space(self) -> Dict[str, Any]:
        """
        ACT 动作空间 (15 维):
        - right_arm/position: 7 (右臂关节)
        - right_gripper/position: 1 (右 gripper)
        - waist/position: 3 (height, pitch, yaw)
        - head/position: 2 (pitch, yaw)
        - base/velocity: 2 (x, yaw)
        """
        return {"shape": (15,), "dtype": "float32", "low": -1.0, "high": 1.0}
    
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        obs = self.get_observation()
        info = {}
        return obs, info
    
    def get_observation(self) -> Dict[str, Any]:
        '''
        Returns:
            qpos: Bx15 - 右臂(7) + 右gripper(1) + waist(3) + head(2) + base(2)
            images: Bx4x3x480x640 - 4个相机图像
        '''
        obs = collections.OrderedDict()
        
        # ========== ACT qpos (15 维) ==========
        qpos = np.concatenate([
            self.joint_state[7:],      # 7: right arm position
            self.gripper_state[1:2],    # 2: right gripper position
            self.waist_state,      # 3: waist position
            self.head_state,       # 2: head position
            self.chassis_state,    # 2: base velocity
        ])
        obs['qpos'] = torch.from_numpy(qpos.astype(np.float32)).unsqueeze(0)
        if torch.cuda.is_available():
            obs['qpos'] = obs['qpos'].cuda()

        # 图像: Dict[str, np.ndarray] → torch.Tensor (num_cam, C, H, W) on GPU
        if self.image_recorder is not None:
            image_dict = self.image_recorder.get_images()
            obs['images'] = self._images_to_tensor(image_dict).unsqueeze(0)
        else:
            obs['images'] = torch.empty(0)
        
        return obs
    
    def _images_to_tensor(self, image_dict: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        将相机图像转换为 tensor
        
        camera_sdk 输出格式（split_stereo=True 时）:
            {'cam_high_0': img, 'cam_high_1': img, 'cam_right_wrist_0': img, 'cam_right_wrist_1': img}
        
        输出: (num_cam, C, H, W) tensor on GPU
        
        注意: 如果某个相机缺失，会用黑图填充并打印警告
        """
        if not image_dict:
            return torch.empty(0)
        
        images = []
        missing_warned = False
        
        # 先获取一个参考图像用于确定目标尺寸
        sample_img = next(iter(image_dict.values()))
        target_shape = sample_img.shape  # (H, W, C)
        
        # camera_sdk 已经分割好了，直接按顺序取
        # 从 'v4l2/cam_high' 提取 'cam_high'，然后加 _0, _1
        for cam_name in self.camera_names:
            pure_name = cam_name.replace('v4l2/', '').replace('rs/', '')
            
            for suffix in ['_0', '_1']:
                key = f"{pure_name}{suffix}"
                if key not in image_dict:
                    # 缺失时用黑图填充
                    if not missing_warned:
                        print(f"[WARN] Camera missing: {key}, available: {list(image_dict.keys())}. Using black placeholder.")
                        missing_warned = True
                    # 用参考图像的尺寸创建黑图
                    img = np.zeros(target_shape, dtype=np.uint8)
                else:
                    img = image_dict[key]
                    # 确保图像尺寸一致（resize 如果不同）
                    if img.shape != target_shape:
                        import cv2
                        img = cv2.resize(img, (target_shape[1], target_shape[0]))
                
                # (H, W, C) → (C, H, W), BGR→RGB, uint8→float32 normalized
                img = np.transpose(img, (2, 0, 1))  # (H, W, C) → (C, H, W)
                img = img.astype(np.float32) / 255.0  # normalize to [0, 1]
                images.append(img)
        
        if not images:
            raise ValueError("No images found")
        
        # Stack and move to GPU
        images_tensor = torch.from_numpy(np.stack(images, axis=0))  # (num_cam, C, H, W)
        if torch.cuda.is_available():
            images_tensor = images_tensor.cuda()
        
        return images_tensor
    
    def get_action(self) -> np.ndarray:
        """获取当前动作（从 ZCM 订阅的 target 构建）"""
        return np.concatenate([
            self.left_action,
            [self.gripper_control[0]],
            self.right_action,
            [self.gripper_control[1]],
            [self.head_target[2], self.head_target[4], self.head_target[5]] if len(self.head_target) >= 6 else [0, 0, 0],
            self.chassis_control,
        ])
    
    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        执行动作
        
        ACT 动作格式 (15 维):
        - [0:7]   right arm position (右臂 7 关节)
        - [7:8]   right gripper position
        - [8:11]  waist position (height, pitch, yaw)
        - [11:13] head position (pitch, yaw)
        - [13:15] base velocity (x, yaw)
        
        注意: PolicyAction IDL 格式:
        - left_joint_action[8] = left_arm[0:7] + left_gripper（保持当前值）
        - right_joint_action[8] = right_arm[0:7] + right_gripper
        - waist[3], head[2]
        """
        a = np.asarray(action, dtype=np.float64).flatten()
        
        # 解析 ACT 动作 (15 维)
        right_arm = a[:8]       # 8: 右臂关节 + gripper
        waist_pos = a[8:11]     # 3: 腰部
        head_pos = a[11:13]     # 2: 头部
        base_vel = a[13:15]     # 2: 底盘（暂不使用）
        
        # 左臂保持当前状态
        left_arm = np.concatenate([self.joint_state[:7], [self.gripper_state[0]]])
        
        # dry_run 模式：只读取观测，不下发动作
        if not self.dry_run:
            # 转换为 PolicyAction IDL 格式
            self.policy_action.operator_type = "policy"
            self.policy_action.left_joint_action = left_arm.tolist()
            self.policy_action.right_joint_action = right_arm.tolist()
            self.policy_action.waist = waist_pos.tolist()
            self.policy_action.head = head_pos.tolist()
            self.policy_action.publiser_name = "h1_robot_env"
            self.policy_action.done = False
            
            # 发布动作到机器人
            self.zcm_node.publish("policy_action", self.policy_action)
            
            # TODO: 底盘速度通过 ChassisControl 发布 (如果需要)
        
        # 调试：每100步打印一次
        if not hasattr(self, '_step_count'):
            self._step_count = 0
        self._step_count += 1
        if self._step_count % 100 == 1:
            mode = "[DRY-RUN]" if self.dry_run else "[PUBLISH]"
            print(f"[H1] {mode} Action: arm={a[:7].round(3)}, grip={a[7].round(3)}, waist={a[8:11].round(3)}, head={a[11:13].round(3)}")
        
        # 获取观测
        obs = self.get_observation()
        
        # HIL 干预检测
        is_intervention = (self.operator_type != "policy")
        
        reward = 0.0
        terminated = False
        truncated = False
        info = {
            "is_intervention": is_intervention,
            "operator_type": self.operator_type,
            "policy_action": a,
        }
        # print(self.operator_type)
        return obs, reward, terminated, truncated, info
    
    def get_buttons(self) -> Tuple[float, float, float, float]:
        """获取 VR 按钮状态"""
        return self.buttons[0], self.buttons[1], self.buttons[2], self.buttons[3]
    
    def move_to_init_pose(self) -> None:
        """发送复位信号（通过 ZCM）"""
        print("[H1RobotEnv] Waiting for reset...")
        while not self.reset_state:
            init_msg = self._system_init_zcmt()
            init_msg.system_init = 1
            self.zcm_node.publish("reset_signal", init_msg)
            time.sleep(0.5)
        self.reset_state = 0
        print("[H1RobotEnv] Reset done")
    
    # ========== 坐标变换工具函数 ==========
    
    @staticmethod
    def _xyz_rpy_to_homogeneous_matrix(xyz, rpy, degrees=False):
        if degrees:
            rpy = np.radians(rpy)
        roll, pitch, yaw = rpy
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr]
        ])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = xyz
        return T
    
    @staticmethod
    def _homogeneous_to_xyz_rpy(T):
        xyz = T[:3, 3]
        R = T[:3, :3]
        
        if not np.isclose(R[2, 0], 1.0):
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = -np.arcsin(R[2, 0])
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            yaw = 0.0
            if R[2, 0] == -1:
                pitch = np.pi / 2
                roll = yaw + np.arctan2(R[0, 1], R[0, 2])
            else:
                pitch = -np.pi / 2
                roll = -yaw + np.arctan2(-R[0, 1], -R[0, 2])
        
        rpy = np.array([roll, pitch, yaw])
        return np.concatenate([xyz, rpy])

    