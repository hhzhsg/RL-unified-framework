"""
H1机器人环境 - 预留接口
"""
from typing import Dict, Any, Optional, Tuple
import numpy as np
from ..base_env import BaseEnv
from core.orchestration import register_env

import time
import collections
from lib_h1_sdk_python import (
    H1Robot,
    MotorControlMode,
    MotorControl,
    EtherCAT_Motor_Index,
    # ArmAction,       # 枚举: LEFT_ARM / RIGHT_ARM
    # ArmActionData,   # 高层动作数据结构(data[6], flag)，
    MotorInformation
)

from .camera_sdk import ImageRecorder

@register_env("h1_robot")
class H1RobotEnv(BaseEnv):
    """H1机器人环境（预留实现）"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.gripper_state = np.zeros(2)  # left gripper + right gripper
        self.waist_state = np.zeros(2)  # pitch + height
        self.left_eef_state = np.zeros(6)
        self.right_eef_state = np.zeros(6)
        self.head_target = np.zeros(6)
        self.joint_state = np.zeros(14)

        self.control_mode = config.control_mode

        # TODO: need to change to match the sdk
        self.sdk_cameras_name = ['cam_rs_0', 'cam_rs_1', 'cam_rs_2']


        self.T_l_init2base = np.array([[ 0.99595273,  0.,         -0.08987855,  0.39210682],
                                       [ 0.,          1.,          0.,          0.18292719],
                                       [ 0.08987855,  0.,          0.99595273,  0.19168143],
                                       [ 0.,          0.,          0.,          1.        ]])
        self.T_r_init2base = np.array([[ 0.99595273,  0.,         -0.08987855,  0.39233398],
                                       [ 0.,          1.,          0.,         -0.18307313],
                                       [ 0.08987855,  0.,          0.99595273,  0.19178583],
                                       [ 0.,          0.,          0.,          1.        ]])

        robot = H1Robot()
        self.camera_names = config.get(
            "camera_names",
            ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
        )

        self.image_recorder = ImageRecorder(self.sdk_cameras_name)

        self.robot = robot

        self.arm_motor_index =  \
              [eval(f'EtherCAT_Motor_Index.MOTOR_LEFT_ARM_{idx+1}') for idx in range(8)]  \
            + [eval(f'EtherCAT_Motor_Index.MOTOR_RIGHT_ARM_{idx+1}') for idx in range(8)]
        
        self.waist_motor_index = [
            EtherCAT_Motor_Index.MOTOR_LIFT, 
            EtherCAT_Motor_Index.MOTOR_WAIST_DOWN, 
            EtherCAT_Motor_Index.MOTOR_WAIST_UP
        ]
        self.head_motor_index = [
            EtherCAT_Motor_Index.MOTOR_HEAD_DOWN, 
            EtherCAT_Motor_Index.MOTOR_HEAD_UP
        ]
        
        # TODO: need to judge if use the control of chassis
        self.motor_index = self.arm_motor_index + self.waist_motor_index + self.head_motor_index
        self.motor_num = len(self.motor_index)


        max_retry = 10
        conut = 0
        while (conut < max_retry) and not robot.isRobotConnected():
            try:
                robot.robot_connect()
            except:
                print("connect failed, trying to connect again.")
                time.sleep(1)
                conut += 1


        if not robot.switchControlMode(MotorControlMode.GRAVITY_COMPENSATION_LEVEL):
            print("switch mode failed"); return
        print("mode =", robot.getCurrentMode())


        if not robot.robot_init():
            print("robot_init failed"); return
        print("robot initialized")


    @property
    def observation_space(self) -> Dict[str, Any]:
        return {"state": {"shape": (self.joint_state.shape + self.gripper_state.shape,), "dtype": "float32"}}
    
    @property
    def action_space(self) -> Dict[str, Any]:
        return {"shape": (self.joint_state + self.gripper_state.shape,), "dtype": "float32", "low": -1.0, "high": 1.0}
    
    def set_robot(self, robot: Any) -> None:
        self.robot = robot
    
    def move_to_init_pose(self):
        target_joint_positions = np.zeros(self.motor_num)
        current_joint_positions = self.get_joint_position()
        
        print("Moving to initial pose...")
        time_duration = 2.0  # seconds
        steps = 1000
        except_index = 16 # exclude waist height motor index
        target_joint_positions[except_index] = current_joint_positions[except_index]
        
        for step in range(steps):
            alpha = (step + 1) / steps
            interpolated_positions = (1 - alpha) * current_joint_positions + alpha * target_joint_positions
            self._set_joint_action(interpolated_positions)
            time.sleep(time_duration / steps)
        
        print("Reached initial pose.")

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # raise NotImplementedError("需要实现具体机器人接口")
    
        self.move_to_init_pose()
        time.sleep(2)
        obs = self.get_observation()

        return obs

    def get_observation(self):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_joint_position()
        
        # TODO: need to judge if use the control of chassis，填充底盘电机
        obs['qpos'] = np.pad(obs['qpos'], (0, 2), mode='constant', constant_values=0)

        origin_image_dict = self.image_recorder.get_images()
        repack_image_dict = {
            self.camera_names[0]: origin_image_dict["cam_rs_0"],
            self.camera_names[1]: origin_image_dict["cam_rs_1"],
            self.camera_names[2]: origin_image_dict["cam_rs_2"],
        }
        obs['images'] = repack_image_dict
        return obs

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        obs = self.get_observation()
        self._set_joint_action(action)
        return obs

    def _set_joint_action(self, action):
        assert len(action) == self.motor_num
        for idx in range(self.motor_num):
            self.send_motor(self.motor_index[idx], pos=action[idx])
    
    def _set_head_joint(self, action):
        head_motor_num = len(self.head_motor_index)
        for idx in range(head_motor_num):
            self.send_motor(self.head_motor_index[idx], pos=action[idx])

    def _set_waist_joint(self, action):
        assert len(action) == len(self.waist_motor_index)
        for i, idx in enumerate(self.waist_motor_index):
            self.send_motor(idx, pos=action[i])

    def get_joint_position(self):
        joint_position = np.zeros(self.motor_num)
        for idx in range(self.motor_num):
            ret, st = self.get_state(self.motor_index[idx])
            joint_position[idx] = st.Position_Actual
        return joint_position
    
    def send_motor(self, idx, pos=None, speed=None, torque=None):
        """
        构造一个 MotorControl 并发送到指定电机:
        - 只设置你给出的字段, 其他保持默认(通常 0)
        - 对底盘轮子使用 speed (速度控制)
        - 对关节使用 pos (位置控制)
        """
        mc = MotorControl()
        if pos is not None:    mc.Position = pos
        if speed is not None:  mc.Speed = speed
        if torque is not None: mc.Torque = torque
        self.robot.setMotorControl_low(idx, mc)

    def get_state(self, idx):
        """
        获取单个电机状态的鲁棒尝试:
        兼容三种可能的绑定签名:
        getMotorState(idx) -> MotorInformation
        getMotorState(idx, out_obj) -> bool0
        getMotorState() -> 序列  (再用 idx 取)
        """
        if not hasattr(self.robot, "getMotorState"):
            return None
        m = self.robot.getMotorState
        # 尝试直接返回对象
        try:
            ret = m(idx)
            if isinstance(ret, MotorInformation):
                return ret
            if isinstance(ret, (list, tuple)):
                try:
                    return ret[idx]
                except Exception:
                    pass
            return ret
        except TypeError:
            pass
        # 尝试 (idx, out)
        try:
            out = MotorInformation()
            ok = m(idx, out)
            if isinstance(ok, MotorInformation):
                return ok
            return out
        except TypeError:
            pass
        # 尝试无参调用
        try:
            ret = m()
            if isinstance(ret, (list, tuple)):
                return ret[idx]
            return ret
        except TypeError:
            return None
