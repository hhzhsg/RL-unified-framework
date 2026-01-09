#!/usr/bin/env python3
"""
VLA-RL Online 训练入口

使用方式:
    python scripts/train_online.py --config config/train_config.yaml --name online_sac_dummy
    
数据流:
    ┌──────────────────────────────────────────────────────────────┐
    │                      Online RL 数据流                         │
    ├──────────────────────────────────────────────────────────────┤
    │                                                              │
    │  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐   │
    │  │  DummyEnv   │────▶│ InferLoop   │────▶│ RolloutBuf  │   │
    │  └─────────────┘     └──────┬──────┘     └──────┬──────┘   │
    │                             │                    │          │
    │                             │ weight             │ sample   │
    │                             │ pull               │          │
    │                             ▼                    ▼          │
    │                      ┌─────────────┐     ┌─────────────┐   │
    │                      │ WeightSync  │◀────│TrainingLoop │   │
    │                      └─────────────┘     └─────────────┘   │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
"""
import sys
from pathlib import Path

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging
import copy
import time
import threading
from typing import Optional

import torch
import numpy as np

from config import (
    Config, load_config_from_yaml, 
    AlgorithmConfig, EnvConfig, TrainingConfig
)
from model import ModelGroup, MLPPolicy, MLPGaussianPolicy
from model.q_network import QNetwork
from buffer import DataHub
from core import TrainingLoop
from core.inference_loop import InferenceLoop
from core.weight_sync import create_weight_sync
from env import create_env, DummyEnv
from algorithm import ALGORITHM_REGISTRY
from data import Batch


def setup_logging(exp_name: str):
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(exp_name)


def create_model_group(config: Config) -> ModelGroup:
    """
    根据配置创建模型组
    
    Online SAC 需要:
    - policy: MLPGaussianPolicy (必须有 sample 方法)
    - q1, q2: QNetwork
    - target_q1, target_q2: 目标 Q 网络
    """
    model_group = ModelGroup()
    
    state_dim = config.env.state_dim
    action_dim = config.env.action_dim
    hidden_dims = config.model.hidden_dims
    
    # ========== 1. 创建 Policy ==========
    # SAC 必须使用 Gaussian Policy
    if config.model.policy_type == "mlp_gaussian":
        policy = MLPGaussianPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            action_space=config.env.action_space,
        )
    elif config.model.policy_type == "mlp":
        # 兼容配置，但警告
        print("[Warning] SAC requires MLPGaussianPolicy, auto-switching...")
        policy = MLPGaussianPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            action_space=config.env.action_space,
        )
    else:
        raise ValueError(f"Unsupported policy type for SAC: {config.model.policy_type}")
    
    model_group.add("policy", policy, frozen=False)
    
    # ========== 2. 创建 Q 网络 ==========
    q1 = QNetwork(state_dim, action_dim, hidden_dims=[256, 256])
    q2 = QNetwork(state_dim, action_dim, hidden_dims=[256, 256])
    model_group.add("q1", q1, frozen=False)
    model_group.add("q2", q2, frozen=False)
    
    # Target Q 网络
    target_q1 = copy.deepcopy(q1)
    target_q2 = copy.deepcopy(q2)
    model_group.add("target_q1", target_q1, frozen=True)
    model_group.add("target_q2", target_q2, frozen=True)
    
    return model_group


def create_data_hub(config: Config) -> DataHub:
    """创建 DataHub（仅 Rollout Buffer）"""
    data_config = getattr(config, 'data', None)
    
    rollout_capacity = 100000
    if data_config and hasattr(data_config, 'rollout_capacity'):
        rollout_capacity = data_config.rollout_capacity
    
    return DataHub(
        demo_paths=None,  # Online 不需要 demo
        rollout_capacity=rollout_capacity,
    )


def warmup_buffer(
    inference_loop: InferenceLoop, 
    num_steps: int,
    logger: logging.Logger
):
    """预热 buffer：收集初始数据"""
    logger.info(f"[Warmup] Collecting {num_steps} initial steps...")
    
    collected = inference_loop.collect_rollout(
        num_steps=num_steps,
        source="rollout"
    )
    
    logger.info(f"[Warmup] Collected {collected} steps")
    return collected


def inference_worker(
    inference_loop: InferenceLoop,
    collect_interval: int,
    steps_per_collect: int,
    stop_event: threading.Event,
    logger: logging.Logger
):
    """
    推理工作线程
    
    持续收集数据，定期更新权重
    """
    total_collected = 0
    
    while not stop_event.is_set():
        # 收集数据
        collected = inference_loop.collect_rollout(
            num_steps=steps_per_collect,
            source="rollout"
        )
        total_collected += collected
        
        # 定期输出统计
        if total_collected % (steps_per_collect * 10) == 0:
            stats = inference_loop.data_hub.rollout_buffer.get_statistics()
            logger.info(f"[Inference] Total collected: {total_collected}, "
                       f"Buffer size: {stats['num_transitions']}")
        
        # 短暂休息，让出 GIL
        time.sleep(0.001)


def training_worker(
    model_group: ModelGroup,
    data_hub: DataHub,
    config: Config,
    weight_sync,
    stop_event: threading.Event,
    logger: logging.Logger
):
    """
    训练工作线程
    
    从 buffer 采样并训练
    """
    from algorithm import create_algorithm
    
    device = config.device
    model_group.to(device)
    
    # 创建算法
    algorithm = create_algorithm(
        config.algorithm.name,
        model_group,
        config.algorithm
    )
    
    batch_size = config.algorithm.batch_size
    total_steps = config.training.stages[0].max_steps
    log_freq = config.training.log_freq
    save_freq = config.training.save_freq
    sync_freq = getattr(config.weight_sync, 'sync_freq', 100)
    
    step = 0
    weight_version = 0
    
    while step < total_steps and not stop_event.is_set():
        # 检查 buffer 是否有足够数据
        if len(data_hub.rollout_buffer) < batch_size:
            time.sleep(0.1)
            continue
        
        # 采样
        batch = data_hub.sample(
            batch_size=batch_size,
            strategy="rollout_only"
        )
        
        # 训练
        batch = batch.to(device)
        metrics = algorithm.train_step(batch)
        step += 1
        
        # 日志
        if step % log_freq == 0:
            logger.info(f"[Training] Step {step}/{total_steps} | {metrics}")
        
        # 同步权重
        if step % sync_freq == 0 and weight_sync:
            weight_version += 1
            state_dict = {"policy": model_group.get("policy").state_dict()}
            weight_sync.push(state_dict, weight_version)
        
        # 保存检查点
        if step % save_freq == 0:
            save_path = Path(config.training.checkpoint_dir) / f"checkpoint_{step}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            model_group.save(str(save_path))
            logger.info(f"[Training] Saved checkpoint to {save_path}")
    
    # 保存最终模型
    final_path = Path(config.training.checkpoint_dir) / "final_policy.pt"
    model_group.save(str(final_path))
    logger.info(f"[Training] Training finished. Final model saved to {final_path}")
    
    stop_event.set()


def main():
    parser = argparse.ArgumentParser(description="VLA-RL Online Training")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to config YAML file")
    parser.add_argument("--name", type=str, required=True,
                       help="Config name to use")
    parser.add_argument("--warmup", type=int, default=1000,
                       help="Warmup steps before training")
    parser.add_argument("--collect-steps", type=int, default=100,
                       help="Steps to collect per iteration")
    args = parser.parse_args()
    
    # 加载配置
    config = load_config_from_yaml(args.config, args.name)
    logger = setup_logging(config.exp_name)
    
    logger.info("=" * 60)
    logger.info(f"VLA-RL Online Training: {config.exp_name}")
    logger.info("=" * 60)
    logger.info(f"Environment: {config.env.name}")
    logger.info(f"State dim: {config.env.state_dim}, Action dim: {config.env.action_dim}")
    logger.info(f"Algorithm: {config.algorithm.name}")
    logger.info(f"Device: {config.device}")
    logger.info("=" * 60)
    
    # 创建组件
    logger.info("[Setup] Creating components...")
    
    # 1. 创建模型组
    model_group = create_model_group(config)
    logger.info(f"[Setup] Models: {model_group.model_names}")
    
    # 2. 创建 DataHub
    data_hub = create_data_hub(config)
    logger.info(f"[Setup] DataHub created, rollout capacity: {data_hub.rollout_buffer.max_size}")
    
    # 3. 创建环境
    env = create_env(config)
    logger.info(f"[Setup] Environment: {type(env).__name__}")
    
    # 4. 创建权重同步器
    weight_sync = create_weight_sync(config.weight_sync.method)
    logger.info(f"[Setup] WeightSync: {config.weight_sync.method}")
    
    # 5. 创建推理配置
    from config import InferenceConfig
    inference_config = InferenceConfig(
        device="cpu",  # 推理用 CPU，训练用 GPU
        deterministic=False,  # SAC 需要探索
    )
    
    # 6. 创建推理循环
    inference_loop = InferenceLoop(
        policy=model_group.get("policy"),
        env=env,
        config=inference_config,
        data_hub=data_hub,
        weight_sync=weight_sync,
    )
    logger.info("[Setup] InferenceLoop created")
    
    # ========== 预热 Buffer ==========
    warmup_buffer(inference_loop, args.warmup, logger)
    
    # ========== 启动训练 ==========
    stop_event = threading.Event()
    
    # 启动推理线程
    inference_thread = threading.Thread(
        target=inference_worker,
        args=(inference_loop, 100, args.collect_steps, stop_event, logger),
        daemon=True
    )
    
    # 启动训练线程
    training_thread = threading.Thread(
        target=training_worker,
        args=(model_group, data_hub, config, weight_sync, stop_event, logger),
        daemon=True
    )
    
    logger.info("[Main] Starting inference and training threads...")
    inference_thread.start()
    training_thread.start()
    
    # 等待训练完成
    try:
        training_thread.join()
    except KeyboardInterrupt:
        logger.info("[Main] Interrupted by user")
        stop_event.set()
    
    inference_thread.join(timeout=5)
    
    logger.info("[Main] Done!")


if __name__ == "__main__":
    main()
