#!/usr/bin/env python
"""
ACT++ HIL 推理脚本

使用预训练 ACT++ 模型进行 rollout + HIL 干预
支持 standalone 模式（本地收集数据）和分布式模式（连接 Learner）

使用方式:
    # Standalone 模式（推荐先用这个测试）
    python run_act_hil.py --standalone

    # Standalone + DRY-RUN（不控制机器人，只测试数据流）
    python run_act_hil.py --standalone --dry-run

    # 连接 Learner 的分布式模式
    python run_act_hil.py --learner-host 192.168.1.100
"""
import sys
import os

# === 必须在所有其他导入之前添加 ACT++ 路径 ===
ACT_PLUS_PLUS_PATH = '/home/robot/pgp/act-plus-plus'
ACT_DETR_PATH = '/home/robot/pgp/act-plus-plus/detr'  # detr 内部依赖 util 模块
if ACT_PLUS_PLUS_PATH not in sys.path:
    sys.path.insert(0, ACT_PLUS_PLUS_PATH)
if ACT_DETR_PATH not in sys.path:
    sys.path.insert(0, ACT_DETR_PATH)

import argparse
import time

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ACTPolicy 需要在 sys.path 设置之后直接从模块导入
from policies.adapters.act_adapter import ACTPolicy
from env.h1_robot import H1ActEnv
from core.runtime.hil_loop import HILActorLoop, HILActorConfig
from core.synchronization.actor_learner import ActorLearnerConfig


def main():
    parser = argparse.ArgumentParser(description="ACT++ HIL 推理")
    
    # 模式选择
    parser.add_argument("--standalone", action="store_true",
                        help="Standalone 模式：不连接 Learner，本地收集数据")
    parser.add_argument("--learner-host", type=str, default="localhost",
                        help="Learner 地址")
    parser.add_argument("--learner-port", type=int, default=50060,
                        help="Learner 端口")
    
    # 环境配置
    parser.add_argument("--dry-run", action="store_true",
                        help="DRY-RUN 模式：不下发动作到机器人")
    parser.add_argument("--use-camera", action="store_true", default=True,
                        help="启用相机（默认启用）")
    parser.add_argument("--no-camera", action="store_true",
                        help="禁用相机（调试用）")
    
    # 模型配置
    parser.add_argument("--ckpt-dir", type=str, 
                        default=os.path.join(os.path.dirname(__file__), "checkpoints"),
                        help="Checkpoint 目录")
    parser.add_argument("--ckpt-name", type=str, default="policy_last.ckpt",
                        help="Checkpoint 文件名")
    
    # 运行配置
    parser.add_argument("--max-steps", type=int, default=10000,
                        help="最大运行步数")
    parser.add_argument("--max-episode-steps", type=int, default=500,
                        help="每个 episode 最大步数")
    parser.add_argument("--save-dir", type=str, default="./collected_data",
                        help="Standalone 模式数据保存目录")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ACT++ HIL 推理")
    print("=" * 60)
    
    # ========== 1. 加载 ACT 策略 ==========
    print("\n[1/3] 加载 ACT++ 模型...")
    policy = ACTPolicy(
        ckpt_dir=args.ckpt_dir,
        ckpt_name=args.ckpt_name,
        device="cuda",
    )
    
    # ========== 2. 初始化环境 ==========
    print("\n[2/3] 初始化 H1 环境...")
    use_camera = args.use_camera and not args.no_camera
    
    env_config = {
        "use_camera": use_camera,
        "dry_run": args.dry_run,
        "zcm_url": os.environ.get("H1_ZCM_URL", "ipcshm"),
        "split_stereo": True,
    }
    
    if use_camera:
        env_config["camera_grpc_target"] = "localhost:50051"
        env_config["camera_names"] = ['v4l2/cam_high', 'v4l2/cam_right_wrist']
    
    env = H1ActEnv(env_config)
    
    if args.dry_run:
        print("⚠️  DRY-RUN 模式：不下发动作")
    
    # 等待相机就绪
    if use_camera and env.image_recorder is not None:
        print("等待相机就绪...")
        for i in range(30):
            time.sleep(0.1)
            images = env.image_recorder.get_images()
            if images:
                print(f"✅ 相机就绪，{len(images)} 个图像")
                break
        else:
            print("⚠️  相机超时")
    
    # ========== 3. 配置 HIL Actor ==========
    print("\n[3/3] 启动 HIL Actor...")
    
    if args.standalone:
        # Standalone 模式
        actor_config = HILActorConfig(
            deterministic=True,
            max_episode_steps=args.max_episode_steps,
            standalone=True,
            standalone_save_dir=args.save_dir,
        )
        actor = HILActorLoop(
            policy_adapter=policy,
            env=env,
            config=actor_config,
            sync_config=None,
            mode="local",
        )
        print("🟢 STANDALONE 模式：本地收集数据")
        print(f"   数据保存目录: {args.save_dir}")
    else:
        # 分布式模式
        sync_config = ActorLearnerConfig(
            learner_host=args.learner_host,
            learner_port=args.learner_port,
        )
        actor_config = HILActorConfig(
            deterministic=True,
            max_episode_steps=args.max_episode_steps,
            weight_sync_freq=10,
            transition_batch_size=1,
            require_initial_weights=False,  # 使用预训练权重
        )
        actor = HILActorLoop(
            policy_adapter=policy,
            env=env,
            config=actor_config,
            sync_config=sync_config,
            mode="grpc",
        )
        print(f"🔵 分布式模式：连接 Learner @ {args.learner_host}:{args.learner_port}")
    
    # ========== 运行 ==========
    print()
    print("=" * 60)
    print("开始运行！VR 操作员可以随时介入控制")
    print("按 Ctrl+C 停止并保存数据")
    print("=" * 60)
    print()
    
    try:
        actor.run(num_steps=args.max_steps, log_freq=50)
    except KeyboardInterrupt:
        print("\n[中断] 正在保存数据...")
    finally:
        actor.cleanup()
        print("\n" + "=" * 60)
        print("运行结束！")
        print(f"统计: {actor.get_statistics()}")
        print("=" * 60)


if __name__ == "__main__":
    main()
