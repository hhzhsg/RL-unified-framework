#!/usr/bin/env python3
"""
离线训练脚本

从 HDF5 演示数据训练策略

Usage:
    python scripts/train_offline.py --config configs/bc_mlp.yaml --name bc_demo
    python scripts/train_offline.py --demo_paths data/*.hdf5 --steps 10000
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config import Config, AlgorithmConfig, load_config_from_yaml
from policy import MLPPolicy
from network import QNetwork
from algorithm import create_algorithm
from buffer import ReplayBuffer
from data import DataHub
from core import ModelGroup, TrainingLoop
from utils import setup_logger, ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Offline Training")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--config_name", type=str, default="default")
    parser.add_argument("--name", type=str, default="offline_exp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--demo_paths", nargs="+", default=[])
    parser.add_argument("--state_dim", type=int, default=16)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[256, 256])
    parser.add_argument("--algorithm", type=str, default="bc", choices=["bc", "td3bc"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    return parser.parse_args()


def build_config(args) -> Config:
    """构建配置"""
    if args.config:
        config = load_config_from_yaml(args.config, args.config_name)
    else:
        config = Config()
    
    config.exp_name = args.name
    config.seed = args.seed
    config.device = args.device
    config.env.state_dim = args.state_dim
    config.env.action_dim = args.action_dim
    config.algorithm.name = args.algorithm
    config.algorithm.lr = args.lr
    config.algorithm.batch_size = args.batch_size
    config.training.total_steps = args.steps
    config.training.batch_size = args.batch_size
    config.training.log_freq = args.log_freq
    config.training.checkpoint_freq = args.save_freq
    config.training.checkpoint_dir = args.save_dir
    
    if args.demo_paths:
        config.data.demo_paths = args.demo_paths
    
    return config


def build_model_group(config: Config) -> ModelGroup:
    """构建模型组"""
    state_dim = config.env.state_dim
    action_dim = config.env.action_dim
    hidden_dims = config.model.hidden_dims
    
    group = ModelGroup()
    
    policy = MLPPolicy(state_dim=state_dim, action_dim=action_dim, hidden_dims=hidden_dims)
    group.add("policy", policy)
    
    if config.algorithm.name == "td3bc":
        q1 = QNetwork(state_dim, action_dim, hidden_dims)
        q2 = QNetwork(state_dim, action_dim, hidden_dims)
        target_policy = MLPPolicy(state_dim, action_dim, hidden_dims)
        target_policy.load_state_dict(policy.state_dict())
        target_q1 = QNetwork(state_dim, action_dim, hidden_dims)
        target_q1.load_state_dict(q1.state_dict())
        target_q2 = QNetwork(state_dim, action_dim, hidden_dims)
        target_q2.load_state_dict(q2.state_dict())
        
        group.add("q1", q1)
        group.add("q2", q2)
        group.add("target_policy", target_policy, frozen=True)
        group.add("target_q1", target_q1, frozen=True)
        group.add("target_q2", target_q2, frozen=True)
    
    return group


def main():
    args = parse_args()
    config = build_config(args)
    
    logger = setup_logger("train")
    logger.info(f"Experiment: {config.exp_name}")
    
    torch.manual_seed(config.seed)
    
    # 构建数据
    logger.info("Loading data...")
    from data import Transition, Observation, RobotState, Action
    import numpy as np
    
    if config.data.demo_paths:
        from buffer import HDF5DemoBuffer
        demo_buffer = HDF5DemoBuffer(demo_paths=config.data.demo_paths, load_images=False)
    else:
        logger.warning("No demo paths, using dummy data")
        demo_buffer = ReplayBuffer(max_size=1000)
        for _ in range(100):
            t = Transition(
                obs=Observation(),
                robot_state=RobotState(raw_state=np.random.randn(config.env.state_dim).astype(np.float32)),
                action=Action(data=np.random.randn(config.env.action_dim).astype(np.float32)),
                reward=0.0,
                next_obs=Observation(),
                next_robot_state=RobotState(raw_state=np.random.randn(config.env.state_dim).astype(np.float32)),
                done=False,
                source="demo",
            )
            demo_buffer.add_transition(t)
    
    data_hub = DataHub()
    data_hub.register_buffer("demo", demo_buffer)
    logger.info(f"Data: {data_hub}")
    
    # 构建模型
    model_group = build_model_group(config)
    model_group.to(config.device)
    logger.info(f"Model: {model_group}")
    
    # 构建算法
    algorithm = create_algorithm(config.algorithm.name, model_group, config.algorithm)
    logger.info(f"Algorithm: {algorithm.name}")
    
    # 训练
    ensure_dir(config.training.checkpoint_dir)
    
    training_loop = TrainingLoop(
        algorithm=algorithm,
        data_hub=data_hub,
        config=config.training,
        device=config.device,
    )
    
    logger.info(f"Training {config.training.total_steps} steps...")
    training_loop.train(
        num_steps=config.training.total_steps,
        sample_strategy="demo_only",
        log_freq=config.training.log_freq,
        checkpoint_freq=config.training.checkpoint_freq,
        checkpoint_path=f"{config.training.checkpoint_dir}/{config.exp_name}",
    )
    
    final_path = f"{config.training.checkpoint_dir}/{config.exp_name}_final.pt"
    model_group.save(final_path)
    logger.info(f"Done. Saved to {final_path}")


if __name__ == "__main__":
    main()
