#!/usr/bin/env python3
"""
在线训练脚本

支持推理/训练分离，使用 SAC 算法

Usage:
    python scripts/train_online.py --name sac_exp --steps 50000
"""
import argparse
import sys
import os
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from config import Config, EnvConfig, AlgorithmConfig
from policy import MLPGaussianPolicy
from network import QNetwork
from algorithm import SAC
from buffer import ReplayBuffer
from env import DummyEnv
from data import DataHub
from core import ModelGroup, TrainingLoop, InferenceLoop, create_weight_sync
from utils import setup_logger, ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Online Training")
    parser.add_argument("--name", type=str, default="online_exp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--state_dim", type=int, default=16)
    parser.add_argument("--action_dim", type=int, default=4)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[256, 256])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--eval_freq", type=int, default=5000)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--sync_mode", type=str, default="shared_memory")
    return parser.parse_args()


def build_model_group(state_dim: int, action_dim: int, hidden_dims: list) -> ModelGroup:
    """构建 SAC 所需的模型组"""
    group = ModelGroup()
    
    # 随机策略
    policy = MLPGaussianPolicy(state_dim, action_dim, hidden_dims)
    group.add("policy", policy)
    
    # Q 网络
    q1 = QNetwork(state_dim, action_dim, hidden_dims)
    q2 = QNetwork(state_dim, action_dim, hidden_dims)
    group.add("q1", q1)
    group.add("q2", q2)
    
    # 目标 Q 网络
    target_q1 = QNetwork(state_dim, action_dim, hidden_dims)
    target_q1.load_state_dict(q1.state_dict())
    target_q2 = QNetwork(state_dim, action_dim, hidden_dims)
    target_q2.load_state_dict(q2.state_dict())
    group.add("target_q1", target_q1, frozen=True)
    group.add("target_q2", target_q2, frozen=True)
    
    return group


def inference_worker(
    policy: MLPGaussianPolicy,
    env: DummyEnv,
    data_hub: DataHub,
    weight_sync,
    warmup_steps: int,
    total_steps: int,
    stop_event: threading.Event,
    logger,
):
    """推理进程"""
    from config import InferenceConfig
    
    inference_loop = InferenceLoop(
        policy=policy,
        env=env,
        config=InferenceConfig(deterministic=False),
        data_hub=data_hub,
        weight_sync=weight_sync,
    )
    
    collected = 0
    while not stop_event.is_set() and collected < total_steps:
        # Warmup 阶段使用随机动作
        deterministic = False
        
        n = inference_loop.collect_rollout(
            num_steps=100,
            source="rollout",
            deterministic=deterministic,
        )
        collected += n
        
        if collected % 1000 == 0:
            logger.info(f"[Inference] Collected {collected} steps, {inference_loop.episode_count} episodes")
    
    logger.info(f"[Inference] Done. Total {collected} steps")


def main():
    args = parse_args()
    
    logger = setup_logger("train")
    logger.info(f"Experiment: {args.name}")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 构建模型
    model_group = build_model_group(args.state_dim, args.action_dim, args.hidden_dims)
    model_group.to(args.device)
    logger.info(f"Model: {model_group}")
    
    # 构建数据中心
    data_hub = DataHub(rollout_capacity=100000)
    
    # 构建环境
    env_config = EnvConfig(state_dim=args.state_dim, action_dim=args.action_dim)
    env = DummyEnv(env_config)
    
    # 权重同步
    weight_sync = create_weight_sync(args.sync_mode)
    
    # 推理用的策略副本
    inference_policy = MLPGaussianPolicy(args.state_dim, args.action_dim, args.hidden_dims)
    inference_policy.load_state_dict(model_group.get("policy").state_dict())
    
    # 启动推理线程
    stop_event = threading.Event()
    inference_thread = threading.Thread(
        target=inference_worker,
        args=(inference_policy, env, data_hub, weight_sync, 
              args.warmup_steps, args.steps * 2, stop_event, logger),
    )
    inference_thread.start()
    
    # 等待 warmup
    logger.info(f"Waiting for {args.warmup_steps} warmup steps...")
    while len(data_hub.rollout_buffer) < args.warmup_steps:
        time.sleep(0.1)
    logger.info(f"Warmup done. Buffer size: {len(data_hub.rollout_buffer)}")
    
    # 构建算法
    algo_config = AlgorithmConfig(name="sac", lr=args.lr, batch_size=args.batch_size)
    algorithm = SAC(model_group, algo_config)
    
    # 训练
    ensure_dir(args.save_dir)
    
    from config import TrainingConfig
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        log_freq=args.log_freq,
    )
    
    training_loop = TrainingLoop(
        algorithm=algorithm,
        data_hub=data_hub,
        config=training_config,
        weight_sync=weight_sync,
        device=args.device,
    )
    
    logger.info(f"Training {args.steps} steps...")
    training_loop.train(
        num_steps=args.steps,
        sample_strategy="rollout_only",
        log_freq=args.log_freq,
        sync_freq=100,
        checkpoint_freq=args.eval_freq,
        checkpoint_path=f"{args.save_dir}/{args.name}",
    )
    
    # 停止推理
    stop_event.set()
    inference_thread.join(timeout=5)
    
    # 评估
    logger.info("Evaluating...")
    from config import InferenceConfig
    eval_loop = InferenceLoop(
        policy=model_group.get("policy"),
        env=DummyEnv(env_config, deterministic=True),
        config=InferenceConfig(deterministic=True),
    )
    eval_results = eval_loop.evaluate(num_episodes=10)
    logger.info(f"Eval: {eval_results}")
    
    # 保存
    final_path = f"{args.save_dir}/{args.name}_final.pt"
    model_group.save(final_path)
    logger.info(f"Done. Saved to {final_path}")


if __name__ == "__main__":
    main()
