#!/usr/bin/env python
"""
使用真正的 HILActorLoop / HILLearnerLoop 进行 gRPC 测试

这个脚本直接使用 core/runtime/hil_loop.py 中的真实 Loop 实现，
验证分布式 HIL 训练的完整流程。

使用方式:
    # 终端 1 - Learner（先启动）
    python test_real_loop.py --role learner --mode grpc --port 50060 --device cpu
    
    # 终端 2 - Actor
    python test_real_loop.py --role actor --mode grpc --learner-host localhost
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 使用框架模块
from policies.adapters import SimpleMLPAdapter, SimpleMLPTrainer
from env.dummy_env import DummyEnv
from core.runtime.hil_loop import HILActorLoop, HILActorConfig, HILLearnerLoop, HILLearnerConfig
from core.synchronization.actor_learner import ActorLearnerConfig


# ============ 主函数 ============

def run_actor(args):
    """运行 Actor"""
    print("=" * 60)
    print("[Actor] Starting with REAL HILActorLoop")
    print("=" * 60)
    
    # 环境（使用框架的 DummyEnv）
    env = DummyEnv(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        max_episode_steps=50,
        intervention_prob=args.intervention_prob,
    )
    print(f"[Actor] Env: DummyEnv ({args.state_dim} → {args.action_dim}, intervention={args.intervention_prob})")
    
    # 策略适配器（使用框架的 SimpleMLPAdapter）
    policy = SimpleMLPAdapter(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        device="cpu",
    )
    print("[Actor] Policy: SimpleMLPAdapter")
    
    # 同步配置
    sync_config = ActorLearnerConfig(
        learner_host=args.learner_host,
        learner_port=args.learner_port,
    )
    
    # Actor 配置
    actor_config = HILActorConfig(
        deterministic=False,
        max_episode_steps=50,
        weight_sync_freq=10,
        transition_batch_size=1,
        require_initial_weights=True,
    )
    
    # 创建 Actor Loop
    actor = HILActorLoop(
        policy_adapter=policy,
        env=env,
        config=actor_config,
        sync_config=sync_config,
        mode=args.mode,
    )
    
    print(f"[Actor] Mode: {args.mode}")
    print(f"[Actor] Learner: {args.learner_host}:{args.learner_port}")
    print()
    
    # 运行
    try:
        results = actor.run(num_steps=args.max_steps, log_freq=100)
        print("\n" + "=" * 60)
        print("[Actor] Finished!")
        print(f"  Stats: {actor.get_statistics()}")
        print("=" * 60)
    finally:
        actor.cleanup()


def run_learner(args):
    """运行 Learner"""
    print("=" * 60)
    print("[Learner] Starting with REAL HILLearnerLoop")
    print("=" * 60)
    
    # 训练适配器（使用框架的 SimpleMLPTrainer）
    trainer = SimpleMLPTrainer(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        lr=args.lr,
        device=args.device,
    )
    print(f"[Learner] Trainer: SimpleMLPTrainer ({args.state_dim} → {args.action_dim})")
    print(f"[Learner] Device: {args.device}")
    
    # 同步配置
    sync_config = ActorLearnerConfig(
        learner_port=args.port,
    )
    
    # Learner 配置
    learner_config = HILLearnerConfig(
        batch_size=args.batch_size,
        utd_ratio=1,
        training_starts=args.training_starts,
        policy_push_frequency=args.weight_push_freq,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        rollout_buffer_capacity=100000,
        intervention_buffer_capacity=50000,
    )
    
    # 创建 Learner Loop
    learner = HILLearnerLoop(
        trainable_adapter=trainer,
        config=learner_config,
        sync_config=sync_config,
        mode=args.mode,
    )
    
    print(f"[Learner] Mode: {args.mode}")
    print(f"[Learner] Port: {args.port}")
    print(f"[Learner] Training starts after: {args.training_starts} transitions")
    print()
    
    # 运行
    try:
        results = learner.run(num_steps=args.max_steps, log_freq=100)
        print("\n" + "=" * 60)
        print("[Learner] Finished!")
        print(f"  Stats: {learner.get_statistics()}")
        print("=" * 60)
    finally:
        learner.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Test Real HIL Loop")
    
    # 角色
    parser.add_argument("--role", type=str, required=True, choices=["actor", "learner"])
    parser.add_argument("--mode", type=str, default="grpc", choices=["local", "grpc"])
    
    # 网络
    parser.add_argument("--learner-host", type=str, default="localhost")
    parser.add_argument("--learner-port", type=int, default=50060)
    parser.add_argument("--port", type=int, default=50060)
    
    # 模型
    parser.add_argument("--state-dim", type=int, default=37)
    parser.add_argument("--action-dim", type=int, default=23)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    
    # 训练
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--training-starts", type=int, default=50, help="开始训练前收集的 transitions 数量")
    parser.add_argument("--weight-push-freq", type=int, default=50)
    parser.add_argument("--checkpoint-freq", type=int, default=200)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/check_hil_loop")
    
    # 环境
    parser.add_argument("--intervention-prob", type=float, default=0.2)
    
    args = parser.parse_args()
    
    if args.role == "actor":
        run_actor(args)
    else:
        run_learner(args)


if __name__ == "__main__":
    main()
