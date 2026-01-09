#!/usr/bin/env python3
"""
VLA-RL 统一训练入口

使用方式:
    python scripts/train.py --config config/train_config.yaml --name offline_bc
    python scripts/train.py --config config/train_config.yaml --name online_sac
"""
import sys
from pathlib import Path

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging
import glob
import torch

from config import (
    Config, load_config_from_yaml, get_data_config,
    AlgorithmConfig, DataSourceConfig
)
from model import ModelGroup, MLPPolicy
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
    """根据配置创建模型组"""
    model_group = ModelGroup()
    
    state_dim = config.env.state_dim
    action_dim = config.env.action_dim
    hidden_dims = config.model.hidden_dims
    
    # 创建 Policy
    if config.model.policy_type == "mlp":
        policy = MLPPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            action_space=config.env.action_space,
        )
        model_group.add("policy", policy, frozen=False)
    
    elif config.model.policy_type == "mlp_gaussian":
        from model import MLPGaussianPolicy
        policy = MLPGaussianPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            action_space=config.env.action_space,
        )
        model_group.add("policy", policy, frozen=False)
    
    else:
        raise ValueError(f"未知的 policy_type: {config.model.policy_type}")
    
    # 如果是 Offline RL 算法，添加 Q 网络
    if config.algorithm.name in ["td3_bc", "cql", "iql"]:
        import copy
        from model.q_network import QNetwork
        
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
    
    return model_group


def create_data_hub(config: Config, data_config: DataSourceConfig) -> DataHub:
    """根据配置创建数据中心"""
    
    # 展开 glob 模式
    demo_paths = []
    for pattern in data_config.demo_paths:
        expanded = glob.glob(pattern, recursive=True)
        demo_paths.extend(expanded)
    
    if data_config.type == "hdf5":
        return DataHub(
            demo_paths=demo_paths if demo_paths else None,
            demo_format="hdf5",
            camera_keys=data_config.camera_keys,
            load_images=data_config.load_images,
            rollout_capacity=data_config.rollout_capacity,
            intervention_capacity=data_config.intervention_capacity,
        )
    else:
        return DataHub(
            demo_paths=None,
            rollout_capacity=data_config.rollout_capacity,
            intervention_capacity=data_config.intervention_capacity,
        )


def main():
    parser = argparse.ArgumentParser(description="VLA-RL 训练")
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
    print("  VLA-RL Training")
    print("=" * 70)
    print(f"\n配置文件: {args.config}")
    print(f"配置名称: {args.name}")
    
    config = load_config_from_yaml(args.config, args.name)
    data_config = get_data_config(config)
    
    # 命令行覆盖
    if args.device:
        config = Config(
            **{k: v for k, v in config.__dict__.items() if k != 'device'},
            device=args.device
        )
    if args.steps:
        config.training.stages[0].max_steps = args.steps
    
    logger = setup_logging(config.exp_name)
    logger.info(f"实验名称: {config.exp_name}")
    logger.info(f"设备: {config.device}")
    logger.info(f"状态维度: {config.env.state_dim}")
    logger.info(f"动作维度: {config.env.action_dim}")
    
    # ========== 2. 创建数据中心 ==========
    logger.info("\n[1/3] 创建数据中心...")
    data_hub = create_data_hub(config, data_config)
    
    stats = data_hub.get_statistics()
    logger.info(f"  Demo Episodes: {stats['demo']['num_episodes']}")
    logger.info(f"  Demo Transitions: {stats['demo']['num_transitions']}")
    
    if stats['demo']['num_transitions'] == 0:
        raise ValueError("没有加载到任何 demo 数据！请检查 data.demo_paths 配置")
    
    # 自动检测维度
    actual_state_dim = stats['demo'].get('state_dim', config.env.state_dim)
    actual_action_dim = stats['demo'].get('action_dim', config.env.action_dim)
    
    if actual_state_dim != config.env.state_dim:
        logger.warning(f"  状态维度不匹配: 配置={config.env.state_dim}, 实际={actual_state_dim}")
        config.env.state_dim = actual_state_dim
    
    if actual_action_dim != config.env.action_dim:
        logger.warning(f"  动作维度不匹配: 配置={config.env.action_dim}, 实际={actual_action_dim}")
        config.env.action_dim = actual_action_dim
    
    # ========== 3. 创建模型 ==========
    logger.info("\n[2/3] 创建模型...")
    model_group = create_model_group(config)
    
    policy = model_group.get("policy")
    num_params = sum(p.numel() for p in policy.parameters())
    logger.info(f"  Policy 类型: {config.model.policy_type}")
    logger.info(f"  Policy 参数量: {num_params:,}")
    
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
    torch.save({
        "policy": policy.state_dict(),
        "config": {
            "exp_name": config.exp_name,
            "state_dim": config.env.state_dim,
            "action_dim": config.env.action_dim,
            "hidden_dims": config.model.hidden_dims,
        },
    }, final_path)
    
    logger.info(f"\n✅ 训练完成!")
    logger.info(f"   模型已保存: {final_path}")
    
    # 训练摘要
    if len(metrics_history) > 10:
        first_losses = [m.get('loss', 0) for m in metrics_history[:5]]
        last_losses = [m.get('loss', 0) for m in metrics_history[-5:]]
        avg_first = sum(first_losses) / len(first_losses)
        avg_last = sum(last_losses) / len(last_losses)
        logger.info(f"\n训练摘要:")
        logger.info(f"   初始 Loss (前5步均值): {avg_first:.4f}")
        logger.info(f"   最终 Loss (后5步均值): {avg_last:.4f}")
        if avg_first > 0:
            logger.info(f"   下降比例: {(avg_first - avg_last) / avg_first * 100:.1f}%")


if __name__ == "__main__":
    main()
