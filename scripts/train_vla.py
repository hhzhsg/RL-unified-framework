#!/usr/bin/env python3
"""
π0/π0.5/RECAP 训练示例脚本

演示如何使用 VLA 模块进行训练:
1. π0 Flow Matching 训练
2. π0.5 离散动作训练  
3. RECAP 3 阶段训练 (π0.6*)

Usage:
    # π0 Flow Matching
    python scripts/train_vla.py --model pi0 --steps 1000
    
    # π0.5 with discrete actions
    python scripts/train_vla.py --model pi05 --steps 1000
    
    # RECAP 3-stage training
    python scripts/train_vla.py --model recap --sft-steps 500 --value-steps 500 --awr-steps 500
"""
import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from model.vla import (
    PI0Config,
    PI05Config,
    ValueConfig,
    RECAPConfig,
    PI0Policy,
    PI05Policy,
    ValueFunction,
)
from algorithm.recap import RECAPAlgorithm, create_recap_models
from model.model_group import ModelGroup


def create_dummy_batch(batch_size: int, config, device: torch.device, num_image_tokens: int = 64):
    """创建虚拟 batch 用于测试"""
    # Dummy images
    images = torch.randn(batch_size, 3, 224, 224, device=device)
    img_masks = torch.ones(batch_size, num_image_tokens, dtype=torch.bool, device=device)
    
    # Dummy language
    lang_tokens = torch.randint(0, 1000, (batch_size, 64), device=device)
    lang_masks = torch.ones(batch_size, 64, dtype=torch.bool, device=device)
    
    # Dummy robot state
    robot_state = torch.randn(batch_size, config.max_state_dim, device=device)
    
    # Dummy actions
    actions = torch.randn(batch_size, config.n_action_steps, config.max_action_dim, device=device)
    
    return {
        "images": [images],
        "img_masks": [img_masks],
        "lang_tokens": lang_tokens,
        "lang_masks": lang_masks,
        "robot_state": robot_state,
        "actions": actions,
    }


def train_pi0(args):
    """训练 π0 Flow Matching 模型"""
    print("=" * 60)
    print("Training π0 Flow Matching Model")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create config - 使用小配置用于测试
    config = PI0Config(
        max_state_dim=32,
        max_action_dim=16,
        n_action_steps=10,  # 减少 action steps
        num_steps=5,  # 减少 denoise steps
        proj_width=256,  # 小维度
    )
    print(f"Config: state_dim={config.max_state_dim}, action_dim={config.max_action_dim}")
    
    # Create model - 使用 tiny 配置以避免 OOM
    model = PI0Policy(config, use_tiny=True).to(device).float()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training loop
    model.train()
    for step in range(args.steps):
        batch = create_dummy_batch(args.batch_size, config, device)
        
        # 确保输入是 float32
        images = [img.float() for img in batch["images"]]
        robot_state = batch["robot_state"].float()
        actions = batch["actions"].float()
        
        # Forward pass (需要传入 state)
        losses = model.model.forward(
            images,
            batch["img_masks"],
            batch["lang_tokens"],
            batch["lang_masks"],
            robot_state,  # state 参数
            actions,
        )
        
        loss = losses["MSE"]
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Step {step}/{args.steps}, MSE Loss: {loss.item():.4f}")
    
    print(f"\nFinal Loss: {loss.item():.4f}")
    print("π0 training complete!")
    
    # Test inference
    model.eval()
    with torch.no_grad():
        actions = model.model.sample_actions(
            images,
            batch["img_masks"],
            batch["lang_tokens"],
            batch["lang_masks"],
            robot_state,  # 添加 state 参数
        )
        print(f"Sampled actions shape: {actions.shape}")


def train_pi05(args):
    """训练 π0.5 模型 (带离散动作)"""
    print("=" * 60)
    print("Training π0.5 Model (with Discrete Actions)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create config - 使用小配置
    config = PI05Config(
        max_state_dim=32,
        max_action_dim=16,
        n_action_steps=10,
        num_steps=5,
        proj_width=256,
        use_knowledge_insulation=True,
        use_adarms=True,
        discrete_action_vocab_size=1000,  # 小 vocab 用于测试
        discrete_action_max_length=32,
    )
    print(f"Config: KI={config.use_knowledge_insulation}, AdaRMS={config.use_adarms}")
    
    # Create model - 使用 tiny 配置
    model = PI05Policy(config, use_tiny=True).to(device).float()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training loop
    model.train()
    for step in range(args.steps):
        batch = create_dummy_batch(args.batch_size, config, device, num_image_tokens=64)
        
        # Add discrete actions
        discrete_actions = torch.randint(
            0, config.discrete_action_vocab_size,
            (args.batch_size, config.discrete_action_max_length),
            device=device
        )
        discrete_action_masks = torch.ones_like(discrete_actions, dtype=torch.bool)
        
        # 确保数据类型一致
        images = [img.float() for img in batch["images"]]
        actions = batch["actions"].float()
        
        # Forward pass
        losses = model.model.forward(
            images,
            batch["img_masks"],
            batch["lang_tokens"],
            batch["lang_masks"],
            actions,
            discrete_actions=discrete_actions,
            discrete_action_masks=discrete_action_masks,
        )
        
        loss = losses["MSE"] + losses["CE"]
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Step {step}/{args.steps}, MSE: {losses['MSE'].item():.4f}, CE: {losses['CE'].item():.4f}")
    
    print(f"\nFinal Loss: MSE={losses['MSE'].item():.4f}, CE={losses['CE'].item():.4f}")
    print("π0.5 training complete!")


def train_recap(args):
    """训练 RECAP (π0.6*) 3 阶段"""
    print("=" * 60)
    print("Training RECAP (π0.6*) - 3-Stage Offline RL")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create configs
    policy_config = PI0Config(
        max_state_dim=32,
        max_action_dim=16,
        n_action_steps=50,
        num_steps=10,
        proj_width=256,
    )
    
    value_config = ValueConfig(
        image_size=224,
        vision_dim=256,
        hidden_dim=256,
        num_layers=4,
        number_of_bins=201,
    )
    
    recap_config = RECAPConfig(
        policy_config=policy_config,
        value_config=value_config,
        sft_steps=args.sft_steps,
        value_steps=args.value_steps,
        awr_steps=args.awr_steps,
        batch_size=args.batch_size,
        policy_lr=1e-4,
        value_lr=1e-4,
        advantage_threshold=0.0,
    )
    
    print(f"RECAP Config: SFT={args.sft_steps}, Value={args.value_steps}, AWR={args.awr_steps}")
    
    # Create models
    model_group = create_recap_models(policy_config, value_config)
    model_group.to(device)
    
    policy = model_group.get("policy")
    value_fn = model_group.get("value")
    
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"Value parameters: {sum(p.numel() for p in value_fn.parameters()):,}")
    
    # Create algorithm
    algorithm = RECAPAlgorithm(recap_config, model_group, device)
    
    # Helper to create batch with returns
    def create_batch_with_returns():
        batch = create_dummy_batch(args.batch_size, policy_config, device)
        
        # Create a simple Batch-like object
        class SimpleBatch:
            pass
        
        b = SimpleBatch()
        b.robot_state = batch["robot_state"]
        b.action = batch["actions"]  # 保留完整的 (B, n_action_steps, action_dim)
        b.reward = torch.rand(args.batch_size, device=device) * 10
        b.returns = b.reward  # 简化: 使用 reward 作为 return
        b.obs = {"observation.image": batch["images"][0]}
        
        return b
    
    # Stage 1: SFT
    print("\n--- Stage 1: SFT ---")
    algorithm.set_stage("sft")
    for step in range(args.sft_steps):
        batch = create_batch_with_returns()
        metrics = algorithm.train_step(batch)
        
        if step % 100 == 0:
            print(f"  Step {step}/{args.sft_steps}: {metrics}")
    
    # Stage 2: Value Training
    print("\n--- Stage 2: Value Training ---")
    algorithm.set_stage("value")
    for step in range(args.value_steps):
        batch = create_batch_with_returns()
        metrics = algorithm.train_step(batch)
        
        if step % 100 == 0:
            print(f"  Step {step}/{args.value_steps}: {metrics}")
    
    # Stage 3: AWR
    print("\n--- Stage 3: AWR ---")
    algorithm.set_stage("awr")
    for step in range(args.awr_steps):
        batch = create_batch_with_returns()
        metrics = algorithm.train_step(batch)
        
        if step % 100 == 0:
            print(f"  Step {step}/{args.awr_steps}: {metrics}")
    
    print("\nRECAP training complete!")


def main():
    parser = argparse.ArgumentParser(description="VLA Model Training")
    parser.add_argument(
        "--model", type=str, default="pi0",
        choices=["pi0", "pi05", "recap"],
        help="Model to train"
    )
    parser.add_argument("--steps", type=int, default=1000, help="Training steps (for pi0/pi05)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--sft-steps", type=int, default=500, help="SFT steps (RECAP)")
    parser.add_argument("--value-steps", type=int, default=500, help="Value training steps (RECAP)")
    parser.add_argument("--awr-steps", type=int, default=500, help="AWR steps (RECAP)")
    
    args = parser.parse_args()
    
    if args.model == "pi0":
        train_pi0(args)
    elif args.model == "pi05":
        train_pi05(args)
    elif args.model == "recap":
        train_recap(args)
    else:
        print(f"Unknown model: {args.model}")
        sys.exit(1)


if __name__ == "__main__":
    main()
