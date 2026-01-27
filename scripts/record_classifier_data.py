#!/usr/bin/env python
"""
Classifier 数据采集脚本

人类专家通过 VR 遥操作 H1 机器人执行任务，在成功时按键标记当前帧。

采集逻辑（参考 HIL-SERL record_success_fail.py）：
1. 人类通过 VR 遥操作机器人执行任务
2. 在任务成功的那一帧，按下指定按键（默认空格）标记为成功样本
3. 其他帧自动标记为失败样本
4. 达到目标成功样本数后保存数据

使用示例:
    # 采集 200 个成功帧
    python scripts/record_classifier_data.py --config projects/h1_hil/config.yaml --successes 200
    
    # 指定成功按键为 's'
    python scripts/record_classifier_data.py --config projects/h1_hil/config.yaml --successes 200 --success_key s
    
    # 使用 DummyEnv 测试（随机标记）
    python scripts/record_classifier_data.py --config projects/_template/config.yaml --successes 10 --test_mode
"""
import argparse
import copy
import sys
import os
import datetime
import pickle as pkl
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Optional

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import load_yaml, Logger

# 导入模块以触发装饰器注册
import env  # noqa: F401

# 键盘监听（可选依赖）
try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("[Warning] pynput not installed, keyboard listening disabled")
    print("[Warning] Install with: pip install pynput")


class KeyboardListener:
    """键盘监听器，检测成功按键"""
    
    def __init__(self, success_key: str = "space"):
        self.success_key = success_key
        self.success_pressed = False
        self.quit_pressed = False
        self.listener = None
        
        if PYNPUT_AVAILABLE:
            self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
    
    def start(self):
        if self.listener:
            self.listener.start()
            print(f"[Keyboard] Listening... Press '{self.success_key}' to mark success, 'q' to quit")
    
    def stop(self):
        if self.listener:
            self.listener.stop()
    
    def _on_press(self, key):
        try:
            # 检测成功按键
            key_str = str(key)
            if self.success_key == "space" and key_str == "Key.space":
                self.success_pressed = True
            elif hasattr(key, 'char') and key.char == self.success_key:
                self.success_pressed = True
            
            # 检测退出按键
            if hasattr(key, 'char') and key.char == 'q':
                self.quit_pressed = True
                
        except AttributeError:
            pass
    
    def _on_release(self, key):
        pass
    
    def check_success(self) -> bool:
        """检查并重置成功标志"""
        if self.success_pressed:
            self.success_pressed = False
            return True
        return False
    
    def check_quit(self) -> bool:
        return self.quit_pressed


def extract_images(obs: Dict[str, Any], camera_names: list) -> Dict[str, np.ndarray]:
    """从观测中提取相机图像"""
    images = {}
    
    # 尝试不同的图像键格式
    if "images" in obs:
        for cam in camera_names:
            if cam in obs["images"]:
                images[cam] = obs["images"][cam]
    else:
        # 直接在 obs 中查找
        for cam in camera_names:
            if cam in obs:
                images[cam] = obs[cam]
    
    return images


def build_transition(
    obs: Dict[str, Any],
    action: np.ndarray,
    next_obs: Dict[str, Any],
    reward: float,
    done: bool,
    camera_names: list,
) -> Dict[str, Any]:
    """构建 transition 字典"""
    transition = {
        "observations": extract_images(obs, camera_names),
        "actions": action.copy(),
        "next_observations": extract_images(next_obs, camera_names),
        "rewards": reward,
        "masks": 1.0 - float(done),
        "dones": done,
    }
    
    # 同时保存状态（如果有）
    if "state" in obs:
        transition["state"] = obs["state"]
    if "qpos" in obs:
        transition["qpos"] = obs["qpos"]
    
    return transition


def main():
    parser = argparse.ArgumentParser(description="Record Classifier Data (HIL-SERL style)")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--successes", type=int, default=200, 
                        help="Number of successful transitions to collect")
    parser.add_argument("--success_key", type=str, default="space",
                        help="Key to press for marking success (default: space)")
    parser.add_argument("--output_dir", type=str, default="./classifier_data",
                        help="Output directory for classifier data")
    parser.add_argument("--cameras", type=str, nargs="+",
                        default=["cam_high", "cam_left_wrist", "cam_right_wrist"],
                        help="Camera names to record")
    parser.add_argument("--test_mode", action="store_true",
                        help="Test mode: randomly mark success (no keyboard)")
    parser.add_argument("--test_success_prob", type=float, default=0.01,
                        help="Success probability in test mode")
    args = parser.parse_args()
    
    # 加载配置
    config = load_yaml(args.config)
    logger = Logger(log_dir="./logs")
    
    # 获取环境配置
    env_config = dict(config.get("env", {}))
    env_type = env_config.pop("type", "dummy")
    
    # 相机配置（优先使用配置文件中的）
    camera_names = env_config.get("camera_names", args.cameras)
    # 如果命令行指定了相机，覆盖配置
    if args.cameras != ["cam_high", "cam_left_wrist", "cam_right_wrist"]:
        camera_names = args.cameras
    
    # Classifier 采集必须启用相机
    if not env_config.get("use_camera", False):
        logger.warning("Classifier data requires camera! Enabling use_camera=True")
        env_config["use_camera"] = True
    
    # 构建环境
    from core.orchestration.component_registry import REGISTRY
    env_cls = REGISTRY.get("env", env_type)
    if env_cls is None:
        raise ValueError(f"Unknown env type: {env_type}")
    robot_env = env_cls(env_config)
    
    logger.info(f"Recording classifier data for: {config.get('project_name', 'unknown')}")
    logger.info(f"Target successes: {args.successes}")
    logger.info(f"Success key: '{args.success_key}'")
    logger.info(f"Cameras: {camera_names}")
    
    # 初始化键盘监听
    kb_listener = KeyboardListener(success_key=args.success_key)
    if not args.test_mode:
        kb_listener.start()
    
    # 采集循环
    successes = []
    failures = []
    
    obs, _ = robot_env.reset()
    pbar = tqdm(total=args.successes, desc="Collecting successes")
    
    try:
        while len(successes) < args.successes:
            # 检查退出
            if kb_listener.check_quit():
                logger.info("Quit requested by user")
                break
            
            # 获取动作（VR 遥操作或环境默认）
            # H1 机器人通过 VR 设备控制，动作从 info["intervene_action"] 获取
            action = np.zeros(robot_env.action_space["shape"])
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = robot_env.step(action)
            done = terminated or truncated
            
            # 检查是否有 VR 遥操作输入
            if "intervene_action" in info:
                action = info["intervene_action"]
            
            # 构建 transition
            transition = build_transition(
                obs=obs,
                action=action,
                next_obs=next_obs,
                reward=reward,
                done=done,
                camera_names=camera_names,
            )
            
            # 检测成功标记
            is_success = False
            if args.test_mode:
                # 测试模式：随机标记
                is_success = np.random.random() < args.test_success_prob
            else:
                # 正常模式：检测按键
                is_success = kb_listener.check_success()
            
            # 分类存储
            if is_success:
                successes.append(copy.deepcopy(transition))
                pbar.update(1)
                pbar.set_postfix({"failures": len(failures)})
            else:
                failures.append(copy.deepcopy(transition))
            
            # 更新状态
            obs = next_obs
            
            # Episode 结束重置
            if done:
                obs, _ = robot_env.reset()
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C)")
    
    finally:
        kb_listener.stop()
        pbar.close()
    
    # 保存数据
    if len(successes) > 0 or len(failures) > 0:
        os.makedirs(args.output_dir, exist_ok=True)
        uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_name = config.get("project_name", "unknown")
        
        # 保存成功样本
        if successes:
            success_file = f"{args.output_dir}/{exp_name}_{len(successes)}_success_{uuid}.pkl"
            with open(success_file, "wb") as f:
                pkl.dump(successes, f)
            logger.info(f"Saved {len(successes)} success transitions to {success_file}")
        
        # 保存失败样本
        if failures:
            failure_file = f"{args.output_dir}/{exp_name}_{len(failures)}_failure_{uuid}.pkl"
            with open(failure_file, "wb") as f:
                pkl.dump(failures, f)
            logger.info(f"Saved {len(failures)} failure transitions to {failure_file}")
        
        # 打印统计
        total = len(successes) + len(failures)
        logger.info(f"Total transitions: {total}")
        logger.info(f"Success ratio: {len(successes) / total:.2%}" if total > 0 else "No data")
    
    else:
        logger.warning("No data collected!")


if __name__ == "__main__":
    main()
