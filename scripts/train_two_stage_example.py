#!/usr/bin/env python3
"""
VLA-RL 两阶段训练：Offline → Online

Stage 1: Offline TD3+BC (Demo Only)
Stage 2: Online SAC (Demo + Rollout 混合，Rollout 初始为空自动退化)

使用方式:
    python scripts/train_two_stage.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import copy
import time
import threading
import logging
import torch
import numpy as np

from config import AlgorithmConfig, EnvConfig, TrainingConfig, StageConfig
from model import ModelGroup, MLPPolicy, MLPGaussianPolicy
from model.q_network import QNetwork
from buffer import DataHub
from algorithm import create_algorithm
from env import DummyEnv
from core.weight_sync import create_weight_sync
from data import Action


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("TwoStage")


def create_model_group(state_dim: int, action_dim: int) -> ModelGroup:
    """创建模型组：Policy + Q1 + Q2 + Target Q"""
    model_group = ModelGroup()
    
    # Policy: 使用 Gaussian 用于 SAC 的熵正则
    policy = MLPGaussianPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256],
    )
    model_group.add("policy", policy)
    
    # Q 网络
    q1 = QNetwork(state_dim, action_dim, hidden_dims=[256, 256])
    q2 = QNetwork(state_dim, action_dim, hidden_dims=[256, 256])
    model_group.add("q1", q1)
    model_group.add("q2", q2)
    
    # Target Q 网络 (冻结)
    target_q1 = copy.deepcopy(q1)
    target_q2 = copy.deepcopy(q2)
    model_group.add("target_q1", target_q1, frozen=True)
    model_group.add("target_q2", target_q2, frozen=True)
    
    return model_group


def generate_fake_demo(data_hub: DataHub, env: DummyEnv, num_episodes: int = 10):
    """生成假的 Demo 数据（用于验证流程）"""
    from data import Transition, Observation, RobotState, Action
    
    logger = logging.getLogger("TwoStage")
    logger.info(f"[Demo] Generating {num_episodes} fake demo episodes...")
    
    total_transitions = 0
    for ep in range(num_episodes):
        env_output = env.reset()
        done = False
        
        while not done:
            # 假装是专家动作：朝目标方向移动
            current_state = env_output.robot_state.raw_state
            target = np.zeros_like(current_state)
            
            # 简单的专家策略：朝目标方向移动
            direction = target - current_state
            expert_action = np.clip(direction[:env.action_dim] * 2.0, -1, 1)
            action = Action(data=expert_action.astype(np.float32))
            
            # 保存当前状态
            prev_obs = env_output.obs
            prev_robot_state = env_output.robot_state
            
            # 执行动作
            env_output = env.step(action)
            done = env_output.done
            
            # 创建 transition 并写入 Demo buffer
            transition = Transition(
                obs=prev_obs,
                robot_state=prev_robot_state,
                action=action,
                reward=env_output.reward,
                next_obs=env_output.obs,
                next_robot_state=env_output.robot_state,
                done=done,
                source="demo",
            )
            
            # 直接写入 demo buffer
            data_hub.demo_buffer.add_transition(transition)
            total_transitions += 1
    
    logger.info(f"[Demo] Generated {total_transitions} demo transitions")


def run_stage1_offline(model_group: ModelGroup, data_hub: DataHub, 
                       device: str, max_steps: int, logger):
    """Stage 1: Offline TD3+BC 训练"""
    logger.info("=" * 60)
    logger.info("Stage 1: Offline TD3+BC (Demo Only)")
    logger.info("=" * 60)
    
    model_group.to(device)
    
    # 创建 TD3+BC 算法
    algo_config = AlgorithmConfig(
        name="td3_bc",
        lr=3e-4,
        batch_size=64,
        gamma=0.99,
        tau=0.005,
    )
    algorithm = create_algorithm("td3_bc", model_group, algo_config)
    
    # 训练循环
    for step in range(1, max_steps + 1):
        # 从 Demo 采样
        batch = data_hub.sample_by_source(batch_size=64, source_weights={"demo": 1.0})
        batch = batch.to(device)
        
        # 训练
        metrics = algorithm.train_step(batch)
        
        if step % 100 == 0:
            logger.info(f"[Stage1] Step {step}/{max_steps} | "
                       f"q_loss: {metrics.get('q_loss', 0):.4f}, "
                       f"policy_loss: {metrics.get('policy_loss', 0):.4f}")
    
    logger.info("[Stage1] Offline training completed!")
    return model_group


def run_stage2_online(model_group: ModelGroup, data_hub: DataHub, env: DummyEnv,
                      device: str, max_steps: int, logger):
    """Stage 2: Online SAC 训练 (Demo + Rollout 混合)"""
    logger.info("=" * 60)
    logger.info("Stage 2: Online SAC (Demo + Rollout Mixed)")
    logger.info("=" * 60)
    
    model_group.to(device)
    policy = model_group.get("policy")
    
    # 创建 SAC 算法
    algo_config = AlgorithmConfig(
        name="sac",
        lr=1e-4,  # finetune 用更小学习率
        batch_size=64,
        gamma=0.99,
        tau=0.005,
    )
    algorithm = create_algorithm("sac", model_group, algo_config)
    
    # 权重同步（用于推理）
    weight_sync = create_weight_sync("shared_memory")
    
    # 推理用 policy 副本
    inference_policy = copy.deepcopy(policy)
    inference_policy.eval()
    
    # 统计
    rollout_count = 0
    episode_rewards = []
    current_episode_reward = 0
    
    # 初始化环境
    env_output = env.reset()
    
    for step in range(1, max_steps + 1):
        # ========== 1. 环境交互（收集 Rollout）==========
        with torch.no_grad():
            state_tensor = torch.FloatTensor(env_output.robot_state.raw_state).unsqueeze(0)
            action_data, _ = inference_policy.sample({}, state_tensor)
            action_data = action_data.squeeze(0).numpy()
        
        action = Action(data=action_data)
        
        # 保存当前状态
        prev_obs = env_output.obs
        prev_robot_state = env_output.robot_state
        
        # 执行动作
        env_output = env.step(action)
        current_episode_reward += env_output.reward
        
        # 创建 transition
        from data import Transition
        transition = Transition(
            obs=prev_obs,
            robot_state=prev_robot_state,
            action=action,
            reward=env_output.reward,
            next_obs=env_output.obs,
            next_robot_state=env_output.robot_state,
            done=env_output.done,
            source="rollout",
        )
        
        # 写入 Rollout buffer
        data_hub.rollout_buffer.add_transition(transition)
        rollout_count += 1
        
        # Episode 结束
        if env_output.done:
            episode_rewards.append(current_episode_reward)
            current_episode_reward = 0
            env_output = env.reset()
        
        # ========== 2. 训练（混合采样）==========
        # 使用 sample_by_source：demo 25%, rollout 75%（rollout 为空时自动退化为纯 demo）
        rollout_size = len(data_hub.rollout_buffer)
        if rollout_size > 0:
            batch = data_hub.sample_by_source(
                batch_size=64, 
                source_weights={"demo": 0.25, "rollout": 0.75}
            )
        else:
            # Rollout 为空，纯 Demo
            batch = data_hub.sample_by_source(
                batch_size=64,
                source_weights={"demo": 1.0}
            )
        batch = batch.to(device)
        
        metrics = algorithm.train_step(batch)
        
        # ========== 3. 同步权重到推理 ==========
        if step % 10 == 0:
            inference_policy.load_state_dict(policy.state_dict())
        
        # ========== 4. 日志 ==========
        if step % 100 == 0:
            rollout_size = len(data_hub.rollout_buffer)
            demo_size = len(data_hub.demo_buffer)
            avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
            
            logger.info(
                f"[Stage2] Step {step}/{max_steps} | "
                f"Rollout: {rollout_size}, Demo: {demo_size} | "
                f"q_loss: {metrics.get('q_loss', 0):.4f}, "
                f"policy_loss: {metrics.get('policy_loss', 0):.4f}, "
                f"avg_reward: {avg_reward:.2f}"
            )
    
    logger.info("[Stage2] Online training completed!")
    logger.info(f"[Stage2] Final avg reward (last 10 eps): {np.mean(episode_rewards[-10:]):.2f}")


def main():
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("VLA-RL Two-Stage Training: Offline → Online")
    logger.info("=" * 60)
    
    # 配置
    state_dim = 16
    action_dim = 4
    device = "cpu"  # 验证用 CPU
    
    # 创建环境
    env_config = EnvConfig(name="dummy", state_dim=state_dim, action_dim=action_dim)
    env = DummyEnv(env_config)
    logger.info(f"[Setup] Environment: DummyEnv (state={state_dim}, action={action_dim})")
    
    # 创建 DataHub
    from buffer import RolloutBuffer, SimpleReplayBuffer
    
    data_hub = DataHub()
    
    # 注册 Demo buffer
    demo_buffer = SimpleReplayBuffer(max_size=100000)
    data_hub.register_dataset("demo", demo_buffer, weight=1.0, source_tag="demo")
    
    # 注册 Rollout buffer  
    rollout_buffer = RolloutBuffer(max_size=10000)
    data_hub.register_dataset("rollout", rollout_buffer, weight=1.0, source_tag="rollout")
    
    logger.info(f"[Setup] DataHub created with datasets: {data_hub.dataset_names}")
    
    # 创建模型组
    model_group = create_model_group(state_dim, action_dim)
    logger.info(f"[Setup] Models: {model_group.model_names}")
    
    # ========== 生成假 Demo 数据 ==========
    generate_fake_demo(data_hub, env, num_episodes=20)
    logger.info(f"[Setup] Demo buffer size: {len(data_hub.demo_buffer)}")
    logger.info(f"[Setup] Rollout buffer size: {len(data_hub.rollout_buffer)} (should be 0)")
    
    # ========== Stage 1: Offline TD3+BC ==========
    model_group = run_stage1_offline(
        model_group, data_hub, device, 
        max_steps=500, 
        logger=logger
    )
    
    # ========== Stage 2: Online SAC ==========
    run_stage2_online(
        model_group, data_hub, env, device,
        max_steps=2000,
        logger=logger
    )
    
    logger.info("=" * 60)
    logger.info("Two-Stage Training Completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
