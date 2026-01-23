#!/usr/bin/env python
"""
Demo 采集脚本（基于 Classifier 自动判定成功）

先训练好 Classifier，再用 Classifier 自动判定任务成功，只保存成功的轨迹。
这个流程同时验证了 Classifier 的准确性。

流程（参考 HIL-SERL record_demos.py）：
1. 加载预训练的 Classifier
2. 人类通过 VR 遥操作执行任务
3. Classifier 自动判定每帧是否成功
4. Episode 结束时，如果成功则保存整条轨迹
5. 达到目标成功轨迹数后保存数据

使用示例:
    # 采集 20 条成功 demo
    python scripts/record_demos.py \
        --config projects/h1_hil/config.yaml \
        --classifier checkpoints/h1_classifier.pt \
        --successes 20
    
    # 使用 DummyEnv 测试（随机 success）
    python scripts/record_demos.py \
        --config projects/_template/config.yaml \
        --successes 5 \
        --test_mode
"""
import argparse
import copy
import sys
import os
import datetime
import pickle as pkl
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, List, Optional

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import load_yaml, Logger

# 导入模块以触发装饰器注册
import env  # noqa: F401


class ClassifierWrapper:
    """
    Classifier 包装器，用于预测任务成功
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        camera_keys: List[str],
        threshold: float = 0.5,
        device: str = "cuda",
    ):
        self.camera_keys = camera_keys
        self.threshold = threshold
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 加载模型
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint.get("config", {})
        
        # 重建模型
        from policies.components.classifiers import RewardClassifier
        self.model = RewardClassifier({
            "num_cameras": config.get("num_cameras", len(camera_keys)),
            "input_shape": (3, config.get("image_size", 84), config.get("image_size", 84)),
            "hidden_dim": 256,
        }).to(self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        self.image_size = config.get("image_size", 84)
        print(f"[Classifier] Loaded from {checkpoint_path}")
        print(f"[Classifier] Cameras: {camera_keys}, threshold: {threshold}")
    
    def predict_success(self, obs: Dict[str, Any]) -> bool:
        """预测当前观测是否为成功状态"""
        images = self._extract_and_preprocess_images(obs)
        if images is None:
            return False
        
        with torch.no_grad():
            # 模型期望 List[Tensor]，每个 Tensor 是一个相机的 batch
            images_list = [img.unsqueeze(0).to(self.device) for img in images]
            probs = self.model.predict_reward(images_list, threshold=self.threshold)
            return probs.item() > 0.5
    
    def _extract_and_preprocess_images(self, obs: Dict) -> Optional[List[torch.Tensor]]:
        """从观测中提取并预处理图像"""
        import torch.nn.functional as F
        
        images = []
        for cam in self.camera_keys:
            img = None
            if "images" in obs and cam in obs["images"]:
                img = obs["images"][cam]
            elif cam in obs:
                img = obs[cam]
            
            if img is None:
                return None
            
            # 预处理
            img = np.array(img)
            if img.ndim == 3 and img.shape[-1] in [1, 3, 4]:
                img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            
            img = torch.from_numpy(img).float()
            if img.max() > 1.0:
                img = img / 255.0
            
            # Resize
            if img.shape[-2:] != (self.image_size, self.image_size):
                img = F.interpolate(
                    img.unsqueeze(0),
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            
            images.append(img)
        
        return images


def build_transition(
    obs: Dict[str, Any],
    action: np.ndarray,
    next_obs: Dict[str, Any],
    reward: float,
    done: bool,
    info: Dict[str, Any],
) -> Dict[str, Any]:
    """构建 transition 字典"""
    return {
        "observations": copy.deepcopy(obs),
        "actions": action.copy(),
        "next_observations": copy.deepcopy(next_obs),
        "rewards": reward,
        "masks": 1.0 - float(done),
        "dones": done,
        "infos": copy.deepcopy(info),
    }


def main():
    parser = argparse.ArgumentParser(description="Record Demos (Classifier-based success detection)")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--classifier", type=str, default=None,
                        help="Classifier checkpoint path (required unless --test_mode)")
    parser.add_argument("--successes", type=int, default=20,
                        help="Number of successful demos to collect")
    parser.add_argument("--output_dir", type=str, default="./demo_data",
                        help="Output directory for demo data")
    parser.add_argument("--cameras", type=str, nargs="+",
                        default=["cam_high", "cam_left_wrist", "cam_right_wrist"],
                        help="Camera names for classifier")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classifier success threshold")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--test_mode", action="store_true",
                        help="Test mode: random success (no classifier)")
    parser.add_argument("--test_success_prob", type=float, default=0.1,
                        help="Success probability per episode in test mode")
    args = parser.parse_args()
    
    # 校验参数
    if not args.test_mode and args.classifier is None:
        parser.error("--classifier is required unless using --test_mode")
    
    # 加载配置
    config = load_yaml(args.config)
    logger = Logger(log_dir="./logs")
    
    # 构建环境
    from core.orchestration.component_registry import REGISTRY
    env_config = dict(config["env"])
    env_type = env_config.pop("type")
    env_cls = REGISTRY.get("env", env_type)
    robot_env = env_cls(env_config)
    
    # 加载 Classifier
    classifier = None
    if not args.test_mode:
        classifier = ClassifierWrapper(
            checkpoint_path=args.classifier,
            camera_keys=args.cameras,
            threshold=args.threshold,
            device=args.device,
        )
    
    exp_name = config.get("project_name", "unknown")
    logger.info(f"Recording demos for: {exp_name}")
    logger.info(f"Target successes: {args.successes}")
    if classifier:
        logger.info(f"Classifier: {args.classifier}")
    
    # 采集循环
    all_transitions = []
    success_count = 0
    trajectory = []
    episode_return = 0.0
    
    obs, info = robot_env.reset()
    logger.info("Environment reset, starting collection...")
    
    pbar = tqdm(total=args.successes, desc="Collecting demos")
    
    try:
        while success_count < args.successes:
            # 获取动作（VR 遥操作或零动作）
            action = np.zeros(robot_env.action_space["shape"])
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = robot_env.step(action)
            done = terminated or truncated
            episode_return += reward
            
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
                info=info,
            )
            trajectory.append(transition)
            
            # 使用 Classifier 预测成功（添加到 info 中）
            if classifier:
                succeed = classifier.predict_success(next_obs)
                info["succeed"] = succeed
            elif args.test_mode:
                # 测试模式：episode 结束时随机判定
                info["succeed"] = done and (np.random.random() < args.test_success_prob)
            
            # 更新进度条
            pbar.set_description(f"Return: {episode_return:.2f}")
            
            obs = next_obs
            
            # Episode 结束处理
            if done:
                if info.get("succeed", False):
                    # 成功：保存整个轨迹
                    for t in trajectory:
                        all_transitions.append(copy.deepcopy(t))
                    success_count += 1
                    pbar.update(1)
                    logger.info(f"✅ Demo {success_count}/{args.successes} saved "
                               f"(length={len(trajectory)}, return={episode_return:.2f})")
                else:
                    logger.info(f"❌ Episode failed (length={len(trajectory)}, return={episode_return:.2f})")
                
                # 重置
                trajectory = []
                episode_return = 0.0
                obs, info = robot_env.reset()
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C)")
    
    finally:
        pbar.close()
    
    # 保存数据
    if all_transitions:
        os.makedirs(args.output_dir, exist_ok=True)
        uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"{args.output_dir}/{exp_name}_{success_count}_demos_{uuid}.pkl"
        
        with open(file_name, "wb") as f:
            pkl.dump(all_transitions, f)
        
        logger.info(f"Saved {success_count} demos ({len(all_transitions)} transitions) to {file_name}")
    else:
        logger.warning("No demos collected!")


if __name__ == "__main__":
    main()
