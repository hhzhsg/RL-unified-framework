      
#!/usr/bin/env python
"""
H1 实机 HIL 测试（配置文件驱动）

完整 Actor-Learner 分布式架构：
- Actor 端: 运行 H1RobotEnv，执行策略推理
- Learner 端: 接收 transitions，更新策略

使用方式:
    # Learner 端（先启动）
    python run_h1_test.py --role learner

    # Actor 端（机器人上）
    python run_h1_test.py --role actor

    # 使用其他配置文件
    python run_h1_test.py --role actor --config my_config.yaml

    # 覆盖配置文件中的参数
    python run_h1_test.py --role actor --dry-run
"""
import sys
import os
import argparse
import time
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from env.h1_robot import H1RobotEnv


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ============ StateEcho Policy（测试用）============

class StateEchoPolicy:
    """
    测试用：从 qpos 提取位置信息作为 action（机器人保持静止）
    
    qpos (37 维): [arm_pos(14), arm_vel(14), gripper(2), waist(3), head(2), base(2)]
    action (23 维): [arm_pos(14), gripper(2), waist(3), head(2), base(2)]
    
    obs 格式（对齐 ACT Plus）:
        - qpos: np.array (37,)
        - images: torch.Tensor (num_cam, C, H, W) on GPU
    """
    
    def __init__(self, state_dim: int = 37, action_dim: int = 23):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._dummy_weights = {"step": torch.zeros(1)}
        self._step = 0
    
    def act(self, obs: dict, deterministic: bool = False) -> np.ndarray:
        self._step += 1
        
        # 每20步打印完整 obs 结构
        if self._step % 20 == 1:
            print(f"\n[DEBUG] step={self._step} | obs keys: {list(obs.keys())}")
            for key, val in obs.items():
                if isinstance(val, np.ndarray):
                    # 检查是否是图像数据（3D或4D数组，值范围0-255或0-1）
                    is_image = len(val.shape) >= 3
                    if is_image:
                        print(f"   {key}: shape={val.shape}, dtype={val.dtype}, "
                              f"min={val.min():.2f}, max={val.max():.2f}, mean={val.mean():.2f}")
                    else:
                        # 检查是否全为0
                        is_all_zero = np.allclose(val, 0)
                        status = "⚠️ ALL ZERO!" if is_all_zero else "✓"
                        print(f"   {key}: shape={val.shape}, dtype={val.dtype}, "
                              f"first5={val.flatten()[:5]} {status}")
                elif isinstance(val, torch.Tensor):
                    # images 是 torch.Tensor
                    print(f"   {key}: torch.Tensor shape={val.shape}, device={val.device}, "
                          f"dtype={val.dtype}")
                else:
                    print(f"  ❓ {key}: type={type(val).__name__}")
        
        # 每200步保存一帧图像用于检查
        if self._step % 200 == 0:
            images = obs.get('images', None)
            if images is not None and isinstance(images, torch.Tensor) and images.numel() > 0:
                import cv2
                save_dir = "./debug_images"
                os.makedirs(save_dir, exist_ok=True)
                # images: (num_cam, C, H, W) float32 [0,1] RGB on GPU
                images_np = images.cpu().numpy()
                for i in range(images_np.shape[0]):
                    img = images_np[i]  # (C, H, W)
                    img = np.transpose(img, (1, 2, 0))  # (H, W, C)
                    img = (img * 255).astype(np.uint8)  # [0,1] -> [0,255]
                    img = img[:, :, ::-1]  # RGB -> BGR for cv2
                    filename = f"{save_dir}/step_{self._step}_cam{i}.jpg"
                    cv2.imwrite(filename, img)
                print(f"[DEBUG] step={self._step} 保存 {images_np.shape[0]} 张图像到 {save_dir}/")
        
        # 获取规范 qpos (37 维) - 对齐 ACT Plus 格式
        qpos = obs.get('qpos', None)
        if qpos is None:
            print(f"[WARN] step={self._step}: qpos is None!")
            return np.zeros(self.action_dim, dtype=np.float32)
        
        # 从 qpos 提取位置信息构建 action (23 维)
        # qpos: [arm_pos(14), arm_vel(14), gripper(2), waist(3), head(2), base(2)]
        # action: [arm_pos(14), gripper(2), waist(3), head(2), base(2)]
        arm_pos = qpos[0:14]
        gripper = qpos[28:30]
        waist = qpos[30:33]
        head = qpos[33:35]
        base = qpos[35:37]
        
        action = np.concatenate([arm_pos, gripper, waist, head, base])
        
        # 每10步打印对比
        if self._step % 10 == 1:
            print(f"[Compare] step={self._step}")
            print(f"  qpos({len(qpos)}): arm_L=[{qpos[0]:.3f},{qpos[1]:.3f},{qpos[2]:.3f},...], "
                  f"arm_R=[{qpos[7]:.3f},{qpos[8]:.3f},{qpos[9]:.3f},...]")
            print(f"  action({len(action)}): arm_L=[{action[0]:.3f},{action[1]:.3f},{action[2]:.3f},...], "
                  f"gripper=[{action[14]:.3f},{action[15]:.3f}], waist=[{action[16]:.3f},{action[17]:.3f},{action[18]:.3f}]")
        
        return action.astype(np.float32)
    
    def get_weights(self) -> dict:
        return self._dummy_weights
    
    def load_weights(self, weights: dict) -> None:
        self._dummy_weights = weights
    
    @property
    def device(self) -> torch.device:
        return torch.device("cpu")


class StateEchoTrainer:
    """
    测试用：打印 batch 统计，区分 rollout/intervention
    
    qpos (37 维): [arm_pos(14), arm_vel(14), gripper(2), waist(3), head(2), base(2)]
    action (23 维): [arm_pos(14), gripper(2), waist(3), head(2), base(2)]
    """
    
    def __init__(self, state_dim: int = 37, action_dim: int = 23, device: str = "cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._device = torch.device(device)
        self._dummy_weights = {"step": torch.zeros(1)}
        self._step = 0
    
    # ========== Policy Protocol 方法 ==========
    
    def act(self, obs: dict, deterministic: bool = False) -> np.ndarray:
        qpos = obs.get('qpos', None)
        if qpos is None:
            return np.zeros(self.action_dim, dtype=np.float32)
        # qpos -> action: 提取位置信息
        arm_pos = qpos[0:14]
        gripper = qpos[28:30]
        waist = qpos[30:33]
        head = qpos[33:35]
        base = qpos[35:37]
        return np.concatenate([arm_pos, gripper, waist, head, base]).astype(np.float32)
    
    def get_weights(self) -> dict:
        return self._dummy_weights
    
    def load_weights(self, weights: dict) -> None:
        self._dummy_weights = weights
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    # ========== Trainer Protocol 方法 ==========
    
    def forward(self, obs: dict) -> torch.Tensor:
        # Dummy forward
        return torch.zeros(1, self.action_dim, device=self._device)
    
    def compute_loss(self, batch: dict):
        # Dummy loss
        loss = torch.tensor(0.0, device=self._device)
        return loss, {"dummy_loss": 0.0}
    
    def get_optimizer(self) -> torch.optim.Optimizer:
        # Dummy optimizer (no params)
        return torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.001)
    
    # ========== 测试用 update 方法 ==========
    
    def update(self, batch: dict) -> dict:
        self._step += 1
        self._dummy_weights["step"] = torch.tensor([self._step])
        
        actions = batch.get('action', batch.get('actions', []))
        is_intervention = batch.get('is_intervention', [])
        
        batch_size = len(actions) if hasattr(actions, '__len__') else 0
        intervention_count = sum(is_intervention) if hasattr(is_intervention, '__len__') else 0
        rollout_count = batch_size - intervention_count
        
        print(f"[Trainer] Step {self._step}: batch={batch_size}, "
              f"rollout={rollout_count}, intervention={intervention_count}")
        
        return {"loss": 0.0, "rollout": rollout_count, "intervention": intervention_count}
    
    def save(self, path: str) -> None:
        """保存模型"""
        torch.save({"step": self._step, "weights": self._dummy_weights}, path)
        print(f"[Trainer] Saved to {path}")
    
    def load(self, path: str) -> None:
        """加载模型"""
        data = torch.load(path)
        self._step = data.get("step", 0)
        self._dummy_weights = data.get("weights", self._dummy_weights)
        print(f"[Trainer] Loaded from {path}")


# ============ Actor / Learner ============

def run_actor(config: dict, args):
    from core.runtime.hil_loop import HILActorLoop, HILActorConfig
    from core.synchronization.actor_learner import ActorLearnerConfig
    
    env_cfg = config.get('env', {})
    policy_cfg = config.get('policy', {})
    hil_cfg = config.get('hil', {})
    actor_cfg = hil_cfg.get('actor', {})
    sync_cfg = config.get('sync', {})
    log_cfg = config.get('logging', {})
    
    print("=" * 60)
    print("[Actor] H1 实机 HIL 测试")
    print("=" * 60)
    
    # 命令行参数覆盖配置
    if args.dry_run:
        env_cfg['dry_run'] = True
    if args.use_camera:
        env_cfg['use_camera'] = True
    if args.use_dummy_env:
        env_cfg['type'] = 'dummy'
    
    # 选择环境
    if env_cfg.get('type') == 'dummy' or args.use_dummy_env:
        from env.dummy_env import DummyEnv
        env = DummyEnv(
            state_dim=policy_cfg.get('state_dim', 23),
            action_dim=policy_cfg.get('action_dim', 21),
            max_episode_steps=50,
            intervention_prob=0.2,
        )
        print("[Actor] 使用 DummyEnv（开发机测试模式）")
    else:
        # 构建 H1RobotEnv 配置
        env_config = {
            "use_camera": env_cfg.get('use_camera', False),
            "dry_run": env_cfg.get('dry_run', False),
            "zcm_url": env_cfg.get('zcm_url', 'ipcshm'),
            "idl_module": env_cfg.get('idl_module', 'idl_python'),
            "idl_python_paths": env_cfg.get('idl_python_paths', []),
        }
        # 相机配置
        if env_config['use_camera']:
            env_config["camera_grpc_target"] = env_cfg.get('camera_grpc_target', 'localhost:50051')
            env_config["camera_names"] = env_cfg.get('camera_names', ['v4l2/cam_high', 'v4l2/cam_right_wrist'])
            env_config["split_stereo"] = env_cfg.get('split_stereo', True)
            
        env = H1RobotEnv(env_config)
        print("[Actor] H1RobotEnv 初始化成功")
        
        # 等待相机视频轨道就绪
        if env_config['use_camera'] and env.image_recorder is not None:
            print("[Actor] 等待相机视频流就绪...")
            for i in range(30):  # 最多等 3 秒
                time.sleep(0.1)
                images = env.image_recorder.get_images()
                if images:
                    print(f"[Actor] ✅ 相机就绪，获取到 {len(images)} 个图像: {list(images.keys())}")
                    for name, img in images.items():
                        print(f"       {name}: shape={img.shape}, dtype={img.dtype}")
                    break
            else:
                print("[Actor] ⚠️ 相机超时，images 仍为空（视频轨道可能未就绪）")
    
    # 策略
    state_dim = policy_cfg.get('state_dim', 37)
    action_dim = policy_cfg.get('action_dim', 23)
    policy = StateEchoPolicy(state_dim=state_dim, action_dim=action_dim)
    print("[Actor] Policy: StateEchoPolicy (state → action)")
    
    # Standalone 模式
    standalone = args.standalone or actor_cfg.get('standalone', False)
    
    if standalone:
        # Standalone 模式：不需要通信配置
        actor_config = HILActorConfig(
            deterministic=actor_cfg.get('deterministic', False),
            max_episode_steps=actor_cfg.get('max_episode_steps', 500),
            standalone=True,
            standalone_save_dir=actor_cfg.get('standalone_save_dir', './standalone_data'),
        )
        actor = HILActorLoop(
            policy=policy,
            env=env,
            config=actor_config,
            sync_config=None,
            mode="local",
        )
        print("[Actor] 🟢 STANDALONE 模式：本地收集数据，不连接 Learner")
    else:
        # 正常模式：连接 Learner
        learner_host = args.learner_host or sync_cfg.get('host', 'localhost')
        learner_port = sync_cfg.get('port', 50060)
        
        sync_config = ActorLearnerConfig(
            learner_host=learner_host,
            learner_port=learner_port,
        )
        
        actor_config = HILActorConfig(
            deterministic=actor_cfg.get('deterministic', False),
            max_episode_steps=actor_cfg.get('max_episode_steps', 500),
            weight_sync_freq=actor_cfg.get('weight_sync_freq', 10),
            transition_batch_size=actor_cfg.get('transition_batch_size', 1),
            require_initial_weights=actor_cfg.get('require_initial_weights', True),
        )
        
        actor = HILActorLoop(
            policy=policy,
            env=env,
            config=actor_config,
            sync_config=sync_config,
            mode="grpc",
        )
        print(f"[Actor] Learner: {learner_host}:{learner_port}")
    if env_cfg.get('dry_run'):
        print("[Actor] ⚠️ DRY-RUN 模式：不下发动作")
    print()
    
    log_freq = log_cfg.get('log_freq', 10)
    max_steps = args.max_steps or actor_cfg.get('max_steps', 10000)
    
    try:
        actor.run(num_steps=max_steps, log_freq=log_freq)
        print("\n" + "=" * 60)
        print("[Actor] Finished!")
        print(f"  Stats: {actor.get_statistics()}")
        print("=" * 60)
    finally:
        actor.cleanup()


def run_learner(config: dict, args):
    from core.runtime.hil_loop import HILLearnerLoop, HILLearnerConfig
    from core.synchronization.actor_learner import ActorLearnerConfig
    
    policy_cfg = config.get('policy', {})
    algo_cfg = config.get('algorithm', {})
    hil_cfg = config.get('hil', {})
    learner_cfg = hil_cfg.get('learner', {})
    sync_cfg = config.get('sync', {})
    log_cfg = config.get('logging', {})
    
    print("=" * 60)
    print("[Learner] H1 实机 HIL 测试")
    print("=" * 60)
    
    # Trainer
    trainer = StateEchoTrainer(
        state_dim=policy_cfg.get('state_dim', 37),
        action_dim=policy_cfg.get('action_dim', 23),
        device=algo_cfg.get('device', 'cpu'),
    )
    print("[Learner] Trainer: StateEchoTrainer")
    
    # 通信配置
    learner_port = sync_cfg.get('port', 50060)
    
    sync_config = ActorLearnerConfig(
        learner_port=learner_port,
    )
    
    learner_config = HILLearnerConfig(
        batch_size=learner_cfg.get('batch_size', 64),
        training_starts=learner_cfg.get('training_starts', 50),
        policy_push_frequency=learner_cfg.get('policy_push_frequency', 50),
        checkpoint_freq=learner_cfg.get('checkpoint_freq', 200),
        checkpoint_dir=learner_cfg.get('checkpoint_dir', './checkpoints/h1_hil'),
        device=algo_cfg.get('device', 'cpu'),
    )
    
    learner = HILLearnerLoop(
        trainer=trainer,
        config=learner_config,
        sync_config=sync_config,
        mode="grpc",
    )
    
    print(f"[Learner] Port: {learner_port}")
    print(f"[Learner] Training starts after {learner_cfg.get('training_starts', 50)} transitions")
    print()
    
    log_freq = log_cfg.get('log_freq', 100)
    max_steps = learner_cfg.get('max_steps', 10000)
    
    try:
        learner.run(num_steps=max_steps, log_freq=log_freq)
        print("\n" + "=" * 60)
        print("[Learner] Finished!")
        print(f"  Stats: {learner.get_statistics()}")
        print("=" * 60)
    finally:
        learner.cleanup()


def main():
    parser = argparse.ArgumentParser(description="H1 实机 HIL 测试（配置文件驱动）")
    
    # 必需参数
    parser.add_argument("--role", type=str, required=True, choices=["actor", "learner"],
                        help="运行角色: actor 或 learner")
    
    # 配置文件
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="配置文件路径 (默认: config.yaml)")
    
    # 可覆盖的参数（命令行优先）
    parser.add_argument("--learner-host", type=str, default=None,
                        help="Learner 地址（覆盖配置文件）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只读观测，不下发动作（覆盖配置文件）")
    parser.add_argument("--use-camera", action="store_true",
                        help="启用相机（覆盖配置文件）")
    parser.add_argument("--use-dummy-env", action="store_true",
                        help="使用 DummyEnv（开发机测试）")
    parser.add_argument("--standalone", action="store_true",
                        help="Standalone 模式：不连接 Learner，本地收集数据")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="最大运行步数（覆盖配置文件）")
    
    args = parser.parse_args()
    
    # 加载配置
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)
    
    if os.path.exists(config_path):
        config = load_config(config_path)
        print(f"[Config] 加载配置: {config_path}")
    else:
        print(f"[Config] 配置文件不存在: {config_path}，使用默认配置")
        config = {}
    
    # 运行
    if args.role == "actor":
        run_actor(config, args)
    else:
        run_learner(config, args)


if __name__ == "__main__":
    main()

    