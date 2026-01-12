#!/usr/bin/env python3
"""
VLA-RL 统一训练入口 (Offline)

使用方式:
    python scripts/train.py --config config/train_config.yaml --name offline_bc
    python scripts/train.py --config config/train_config.yaml --name offline_td3bc
"""
import sys
from pathlib import Path

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging
import glob
import copy
import torch

from config import (
    Config, load_config_from_yaml, get_data_config,
    AlgorithmConfig, DataSourceConfig
)
from model import ModelGroup, MLPPolicy, MLPGaussianPolicy
from model.q_network import QNetwork
from buffer import DataHub
from core import TrainingLoop
from algorithm import ALGORITHM_REGISTRY


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
    
    统一命名规范:
    - policy: 策略网络
    - q1, q2: Q 网络 (用于 SAC, TD3+BC, CQL, IQL)
    - target_q1, target_q2: 目标 Q 网络 (frozen)
    - vf: 价值函数 (用于 RECAP/AWR)
    """
    model_group = ModelGroup()
    
    state_dim = config.env.state_dim
    action_dim = config.env.action_dim
    hidden_dims = config.model.hidden_dims
    
    # ========== 1. 创建 Policy ==========
    if config.model.policy_type == "mlp":
        policy = MLPPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            action_space=config.env.action_space,
        )
    elif config.model.policy_type == "mlp_gaussian":
        policy = MLPGaussianPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            action_space=config.env.action_space,
        )
    else:
        raise ValueError(f"未知的 policy_type: {config.model.policy_type}")
    
    model_group.add("policy", policy, frozen=False)
    
    # ========== 2. 根据算法创建额外网络 ==========
    algo_name = config.algorithm.name
    
    # 需要 Q 网络的算法
    q_network_algos = ["td3_bc", "cql", "iql", "sac"]
    
    if algo_name in q_network_algos:
        # 双 Q 网络 (减少过估计)
        q1 = QNetwork(state_dim, action_dim, hidden_dims=[256, 256])
        q2 = QNetwork(state_dim, action_dim, hidden_dims=[256, 256])
        model_group.add("q1", q1, frozen=False)
        model_group.add("q2", q2, frozen=False)
        
        # Target Q 网络 (延迟更新，稳定训练)
        target_q1 = copy.deepcopy(q1)
        target_q2 = copy.deepcopy(q2)
        model_group.add("target_q1", target_q1, frozen=True)
        model_group.add("target_q2", target_q2, frozen=True)
    
    # 需要 V 网络的算法 (IQL, AWR)
    v_network_algos = ["iql", "awr", "vf_regression"]
    
    if algo_name in v_network_algos:
        from model.q_network import VNetwork
        vf = VNetwork(state_dim, hidden_dims=[256, 256])
        model_group.add("vf", vf, frozen=False)
    
    return model_group


def create_data_hub(config: Config, data_config: DataSourceConfig) -> DataHub:
    """根据配置创建数据中心 (新 API: register_dataset)"""
    from buffer import RolloutBuffer, SimpleReplayBuffer
    from buffer.hdf5_buffer import HDF5DemoBuffer
    
    # 展开 glob 模式
    demo_paths = []
    for pattern in data_config.demo_paths:
        expanded = glob.glob(pattern, recursive=True)
        demo_paths.extend(expanded)
    
    # 创建 DataHub
    data_hub = DataHub()
    
    if data_config.type == "hdf5" and demo_paths:
        # HDF5 Demo Buffer
        demo_buffer = HDF5DemoBuffer(
            demo_paths=demo_paths,
            camera_keys=data_config.camera_keys if data_config.camera_keys else [],
            load_images=data_config.load_images,
        )
        data_hub.register_dataset("demo", demo_buffer, weight=1.0, source_tag="demo")
    elif demo_paths:
        # 普通 Demo Buffer
        demo_buffer = SimpleReplayBuffer(max_size=100000)
        data_hub.register_dataset("demo", demo_buffer, weight=1.0, source_tag="demo")
    
    # Rollout Buffer (用于在线数据)
    rollout_capacity = getattr(data_config, 'rollout_capacity', 100000)
    rollout_buffer = RolloutBuffer(max_size=rollout_capacity)
    data_hub.register_dataset("rollout", rollout_buffer, weight=0.0, source_tag="rollout")
    
    return data_hub


def validate_config(config: Config):
    """
    验证配置一致性
    
    检查:
    - SAC 需要 GaussianPolicy
    - 算法名与 stage.algorithm 一致
    """
    algo_name = config.algorithm.name
    policy_type = config.model.policy_type
    
    # SAC 需要 GaussianPolicy
    if algo_name == "sac" and policy_type != "mlp_gaussian":
        raise ValueError(
            f"SAC 需要 policy_type='mlp_gaussian'，当前是 '{policy_type}'"
        )
    
    # 检查 stage 算法与主算法一致
    for stage in config.training.stages:
        if stage.algorithm not in ALGORITHM_REGISTRY:
            raise ValueError(
                f"Stage '{stage.name}' 使用未知算法 '{stage.algorithm}'。"
                f"可用: {list(ALGORITHM_REGISTRY.keys())}"
            )


def main():
    parser = argparse.ArgumentParser(description="VLA-RL 离线训练")
    parser.add_argument("--config", type=str, default="config/train_config.yaml",
                        help="配置文件路径")
    parser.add_argument("--name", type=str, default="offline_bc",
                        help="配置名称")
    parser.add_argument("--device", type=str, default=None,
                        help="覆盖设备配置")
    parser.add_argument("--steps", type=int, default=None,
                        help="覆盖训练步数")
    args = parser.parse_args()
    
    # ========== 1. 加载配置 ==========
    print("=" * 70)
    print("  VLA-RL Offline Training")
    print("=" * 70)
    print(f"\n配置文件: {args.config}")
    print(f"配置名称: {args.name}")
    
    config = load_config_from_yaml(args.config, args.name)
    data_config = get_data_config(config)
    
    # 命令行覆盖
    if args.device:
        config.device = args.device
    if args.steps:
        config.training.stages[0].max_steps = args.steps
    
    # 验证配置
    validate_config(config)
    
    logger = setup_logging(config.exp_name)
    logger.info(f"实验名称: {config.exp_name}")
    logger.info(f"算法: {config.algorithm.name}")
    logger.info(f"设备: {config.device}")
    logger.info(f"状态维度: {config.env.state_dim}")
    logger.info(f"动作维度: {config.env.action_dim}")
    
    # ========== 2. 创建数据中心 ==========
    logger.info("\n[1/3] 创建数据中心...")
    data_hub = create_data_hub(config, data_config)
    
    stats = data_hub.get_statistics()
    demo_stats = stats["datasets"].get("demo", {})
    num_transitions = demo_stats.get("size", 0)
    
    logger.info(f"  Demo Transitions: {num_transitions}")
    logger.info(f"  Total Datasets: {stats['num_datasets']}")
    
    if num_transitions == 0:
        raise ValueError("没有加载到任何 demo 数据！请检查 data.demo_paths 配置")
    
    # 从 HDF5 Buffer 获取维度信息
    demo_dataset = data_hub.get_dataset("demo")
    if demo_dataset and hasattr(demo_dataset.buffer, 'get_statistics'):
        buffer_stats = demo_dataset.buffer.get_statistics()
        actual_state_dim = buffer_stats.get('state_dim', config.env.state_dim)
        actual_action_dim = buffer_stats.get('action_dim', config.env.action_dim)
    else:
        actual_state_dim = config.env.state_dim
        actual_action_dim = config.env.action_dim
    
    if actual_state_dim != config.env.state_dim:
        logger.warning(f"  状态维度不匹配: 配置={config.env.state_dim}, 实际={actual_state_dim}")
        config.env.state_dim = actual_state_dim
    
    if actual_action_dim != config.env.action_dim:
        logger.warning(f"  动作维度不匹配: 配置={config.env.action_dim}, 实际={actual_action_dim}")
        config.env.action_dim = actual_action_dim
    
    # ========== 3. 创建模型 ==========
    logger.info("\n[2/3] 创建模型...")
    model_group = create_model_group(config)
    
    # 打印模型摘要
    summary = model_group.summary()
    for name, info in summary.items():
        logger.info(f"  {name}: {info['num_params']:,} params, frozen={info['frozen']}")
    
    # ========== 4. 创建训练循环并运行 ==========
    logger.info("\n[3/3] 开始训练...")
    
    training_loop = TrainingLoop(
        model_group=model_group,
        data_hub=data_hub,
        config=config.training,
        algo_config=config.algorithm,
        device=config.device,
    )
    
    # 记录训练指标
    metrics_history = []
    
    def callback(step: int, metrics: dict):
        metrics_history.append({"step": step, **metrics})
    
    # 运行训练
    training_loop.run(callback=callback)
    
    # ========== 5. 保存模型 ==========
    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    final_path = checkpoint_dir / "final_policy.pt"
    
    # 保存完整 model_group
    save_data = {
        "model_group": model_group.state_dict(),
        "config": {
            "exp_name": config.exp_name,
            "algorithm": config.algorithm.name,
            "state_dim": config.env.state_dim,
            "action_dim": config.env.action_dim,
            "hidden_dims": config.model.hidden_dims,
            "policy_type": config.model.policy_type,
        },
    }
    torch.save(save_data, final_path)
    
    logger.info(f"\n✅ 训练完成!")
    logger.info(f"   模型已保存: {final_path}")
    
    # 训练摘要
    if len(metrics_history) > 10:
        first_losses = [m.get('loss', m.get('q_loss', 0)) for m in metrics_history[:5]]
        last_losses = [m.get('loss', m.get('q_loss', 0)) for m in metrics_history[-5:]]
        avg_first = sum(first_losses) / len(first_losses) if first_losses else 0
        avg_last = sum(last_losses) / len(last_losses) if last_losses else 0
        
        logger.info(f"\n训练摘要:")
        logger.info(f"   初始 Loss (前5步均值): {avg_first:.4f}")
        logger.info(f"   最终 Loss (后5步均值): {avg_last:.4f}")
        if avg_first > 0:
            reduction = (avg_first - avg_last) / avg_first * 100
            logger.info(f"   下降比例: {reduction:.1f}%")


if __name__ == "__main__":
    main()
