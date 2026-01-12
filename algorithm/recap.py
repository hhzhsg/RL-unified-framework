"""
RECAP Algorithm for π0.6*

Paper: RECAP: Robotic Embodied CAlibratedPolicy
https://www.physicalintelligence.company/download/recap.pdf

RECAP 是一个 3 阶段的离线 RL 算法:

Stage 1: SFT (Supervised Fine-Tuning)
  - 用所有演示数据做标准 BC 训练
  - 训练 π0/π0.5 模型

Stage 2: Value Function Training
  - 用演示数据训练 Value Function
  - Value: V(s) = E[R | s], 离散化为 201 bins
  - 使用 Cross-Entropy loss

Stage 3: AWR (Advantage-Weighted Regression)
  - 计算每个 transition 的 advantage: A(s) = R(s) - V(s)
  - 使用 advantage indicator: I_t = 1 if A(s_t) > threshold
  - 仅用高 advantage 样本训练 policy
  - Loss: L = I_t * L_flow_matching

核心思想: 用 value function 识别"好"的 transition，然后重点学习这些 transition
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from algorithm.base_algorithm import BaseAlgorithm
from data import Batch
from model.model_group import ModelGroup
from model.vla.configuration import RECAPConfig, PI0Config, PI05Config, ValueConfig
from model.vla.pi0_policy import PI0Policy
from model.vla.pi05_policy import PI05Policy
from model.vla.value_function import ValueFunction, compute_n_step_returns, compute_advantage_indicator


class RECAPAlgorithm(BaseAlgorithm):
    """
    RECAP: Robotic Embodied CAlibratedPolicy
    
    3 阶段离线 RL:
    1. SFT: 标准 BC 训练
    2. Value Training: 训练 Value Function
    3. AWR: Advantage-Weighted Regression
    
    Training Flow:
    ```
    ┌─────────────────────────────────────────────────────────────────────┐
    │                      RECAP Training Pipeline                         │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │  Stage 1: SFT                                                       │
    │  ┌──────────────────┐                                               │
    │  │ Demo Data        │──▶ L_flow_matching ──▶ Update Policy          │
    │  └──────────────────┘                                               │
    │                                                                     │
    │  Stage 2: Value Training                                            │
    │  ┌──────────────────┐    ┌──────────────┐                          │
    │  │ Demo Data        │───▶│ Compute      │──▶ L_CE ──▶ Update V(s)  │
    │  │ + Returns        │    │ N-step Return│                          │
    │  └──────────────────┘    └──────────────┘                          │
    │                                                                     │
    │  Stage 3: AWR                                                       │
    │  ┌──────────────────┐    ┌──────────────┐    ┌──────────────────┐  │
    │  │ Demo Data        │───▶│ Compute      │───▶│ Filter by        │  │
    │  │ + Returns        │    │ Advantage    │    │ I_t = A_t > θ    │  │
    │  └──────────────────┘    └──────────────┘    └────────┬─────────┘  │
    │                                                       │            │
    │                                              I_t * L_flow ──▶ Policy│
    └─────────────────────────────────────────────────────────────────────┘
    ```
    """
    
    REQUIRED_MODELS = ["policy", "value"]
    
    def __init__(
        self,
        config: RECAPConfig,
        model_group: ModelGroup,
        device: torch.device = None,
    ):
        # 使用 config 作为 AlgorithmConfig (duck typing)
        super().__init__(model_group, config)
        
        self.config = config
        self.model_group = model_group
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get models from group
        self.policy = model_group.get("policy")
        self.value_fn = model_group.get("value")
        
        # 当前训练阶段
        self.current_stage = "sft"  # "sft", "value", "awr"
        self._step_count = 0
        
        # Setup optimizers
        self._setup_optimizers()
    
    def _setup_optimizers(self):
        """设置优化器"""
        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.config.policy_lr,
            weight_decay=self.config.weight_decay,
        )
        
        self.value_optimizer = torch.optim.AdamW(
            self.value_fn.parameters(),
            lr=self.config.value_lr,
            weight_decay=self.config.weight_decay,
        )
    
    def set_stage(self, stage: str):
        """设置当前训练阶段"""
        assert stage in ["sft", "value", "awr"], f"Unknown stage: {stage}"
        self.current_stage = stage
        print(f"[RECAP] Switched to stage: {stage}")
    
    def train_step(self, batch: Batch) -> Dict[str, float]:
        """
        根据当前阶段执行训练
        
        Args:
            batch: 训练 batch
            
        Returns:
            训练 metrics
        """
        self._step_count += 1
        
        if self.current_stage == "sft":
            return self._train_sft(batch)
        elif self.current_stage == "value":
            return self._train_value(batch)
        elif self.current_stage == "awr":
            return self._train_awr(batch)
        else:
            raise ValueError(f"Unknown stage: {self.current_stage}")
    
    def _train_sft(self, batch: Batch) -> Dict[str, float]:
        """
        Stage 1: Supervised Fine-Tuning
        
        标准 BC 训练，使用所有演示数据
        """
        self.policy.train()
        
        # Prepare inputs
        obs = self._prepare_obs(batch)
        robot_state = batch.robot_state.to(self.device)
        actions = batch.action.to(self.device)
        
        # Forward pass
        if hasattr(self.policy, 'compute_loss'):
            # π0/π0.5 style loss
            losses = self.policy.compute_loss(obs, robot_state, actions)
            loss = losses.get("MSE", losses.get("loss", 0)) + losses.get("CE", 0)
        else:
            # Standard BC
            pred_actions = self.policy.forward(obs, robot_state)
            loss = F.mse_loss(pred_actions, actions)
        
        # Backward
        self.policy_optimizer.zero_grad()
        loss.backward()
        
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip)
        
        self.policy_optimizer.step()
        
        return {
            "sft/loss": loss.item(),
            "sft/mse": losses.get("MSE", loss).item() if isinstance(losses, dict) else loss.item(),
        }
    
    def _train_value(self, batch: Batch) -> Dict[str, float]:
        """
        Stage 2: Value Function Training
        
        使用 n-step returns 训练 value function
        """
        self.value_fn.train()
        
        # Prepare inputs
        images = self._get_images(batch)
        lang_tokens, lang_mask = self._get_language(batch)
        
        # Get returns (预计算或实时计算)
        if hasattr(batch, 'returns') and batch.returns is not None:
            returns = batch.returns.to(self.device)
        else:
            # 使用 reward 作为简化的 return
            returns = batch.reward.to(self.device)
        
        # Forward pass
        output = self.value_fn.forward(images, lang_tokens, lang_mask, returns)
        loss = output["loss"]
        
        # Backward
        self.value_optimizer.zero_grad()
        loss.backward()
        
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.value_fn.parameters(), self.config.grad_clip)
        
        self.value_optimizer.step()
        
        return {
            "value/loss": loss.item(),
        }
    
    def _train_awr(self, batch: Batch) -> Dict[str, float]:
        """
        Stage 3: Advantage-Weighted Regression
        
        只用高 advantage 样本训练 policy
        """
        self.policy.train()
        self.value_fn.eval()
        
        # Prepare inputs
        obs = self._prepare_obs(batch)
        robot_state = batch.robot_state.to(self.device)
        actions = batch.action.to(self.device)
        
        images = self._get_images(batch)
        lang_tokens, lang_mask = self._get_language(batch)
        
        # Get returns
        if hasattr(batch, 'returns') and batch.returns is not None:
            returns = batch.returns.to(self.device)
        else:
            returns = batch.reward.to(self.device)
        
        # Compute advantage
        with torch.no_grad():
            advantages = self.value_fn.compute_advantage(
                images, lang_tokens, lang_mask, returns
            )
        
        # Compute indicator I_t = 1 if A_t > threshold
        indicator = (advantages > self.config.advantage_threshold).float()
        
        # Forward pass
        if hasattr(self.policy, 'compute_loss'):
            losses = self.policy.compute_loss(obs, robot_state, actions)
            mse_loss = losses.get("MSE", losses.get("loss", 0))
            ce_loss = losses.get("CE", 0)
        else:
            pred_actions = self.policy.forward(obs, robot_state)
            mse_loss = F.mse_loss(pred_actions, actions, reduction='none').mean(dim=-1)
            ce_loss = torch.zeros_like(mse_loss)
        
        # Apply indicator weighting
        if mse_loss.dim() > 0:
            weighted_mse = (indicator * mse_loss).mean()
        else:
            weighted_mse = indicator.mean() * mse_loss
        
        loss = weighted_mse + ce_loss if isinstance(ce_loss, Tensor) else weighted_mse
        
        # Backward
        self.policy_optimizer.zero_grad()
        loss.backward()
        
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip)
        
        self.policy_optimizer.step()
        
        return {
            "awr/loss": loss.item(),
            "awr/advantage_mean": advantages.mean().item(),
            "awr/indicator_ratio": indicator.mean().item(),
        }
    
    def _prepare_obs(self, batch: Batch) -> Dict[str, Tensor]:
        """准备观测输入"""
        obs = {}
        
        # Images
        if hasattr(batch, 'obs') and batch.obs is not None:
            if isinstance(batch.obs, dict):
                for k, v in batch.obs.items():
                    if isinstance(v, Tensor):
                        obs[k] = v.to(self.device)
            elif isinstance(batch.obs, Tensor):
                obs["observation.image"] = batch.obs.to(self.device)
        
        # Language
        if hasattr(batch, 'language') and batch.language is not None:
            obs["language"] = batch.language
        
        return obs
    
    def _get_images(self, batch: Batch) -> Tensor:
        """获取图像 tensor"""
        if hasattr(batch, 'obs') and batch.obs is not None:
            if isinstance(batch.obs, dict):
                for k, v in batch.obs.items():
                    if "image" in k.lower():
                        return v.to(self.device)
            elif isinstance(batch.obs, Tensor):
                return batch.obs.to(self.device)
        
        # Placeholder
        return torch.zeros(batch.robot_state.shape[0], 3, 224, 224, device=self.device)
    
    def _get_language(self, batch: Batch) -> Tuple[Tensor, Tensor]:
        """获取语言 tokens"""
        bsize = batch.robot_state.shape[0]
        
        # Placeholder
        tokens = torch.zeros(bsize, 64, dtype=torch.long, device=self.device)
        mask = torch.ones(bsize, 64, dtype=torch.bool, device=self.device)
        
        return tokens, mask


def create_recap_models(
    policy_config: PI0Config | PI05Config,
    value_config: ValueConfig,
) -> ModelGroup:
    """
    创建 RECAP 所需的模型
    
    Args:
        policy_config: Policy 配置 (PI0 或 PI05)
        value_config: Value Function 配置
        
    Returns:
        包含 policy 和 value 的 ModelGroup
    """
    # Create policy
    if isinstance(policy_config, PI05Config):
        policy = PI05Policy(policy_config, use_tiny=True)
    else:
        policy = PI0Policy(policy_config, use_tiny=True)
    
    # Create value function
    value_fn = ValueFunction(value_config)
    
    # Create model group
    model_group = ModelGroup()
    model_group.add("policy", policy)
    model_group.add("value", value_fn)
    
    return model_group


class RECAPTrainer:
    """
    RECAP 训练器
    
    管理 3 阶段训练流程
    """
    
    def __init__(
        self,
        config: RECAPConfig,
        model_group: ModelGroup,
        data_hub,
        device: torch.device = None,
    ):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.algorithm = RECAPAlgorithm(config, model_group, device)
        self.data_hub = data_hub
        
        self._total_steps = 0
    
    def train(self):
        """运行完整的 3 阶段训练"""
        print("[RECAP] Starting 3-stage training")
        
        # Stage 1: SFT
        print(f"\n[RECAP] Stage 1: SFT ({self.config.sft_steps} steps)")
        self.algorithm.set_stage("sft")
        self._run_stage(self.config.sft_steps)
        
        # Stage 2: Value Training
        print(f"\n[RECAP] Stage 2: Value Training ({self.config.value_steps} steps)")
        self.algorithm.set_stage("value")
        self._run_stage(self.config.value_steps)
        
        # Stage 3: AWR
        print(f"\n[RECAP] Stage 3: AWR ({self.config.awr_steps} steps)")
        self.algorithm.set_stage("awr")
        self._run_stage(self.config.awr_steps)
        
        print("\n[RECAP] Training complete!")
    
    def _run_stage(self, num_steps: int):
        """运行单个阶段"""
        for step in range(num_steps):
            batch = self.data_hub.sample(self.config.batch_size)
            batch = batch.to(self.device)
            
            metrics = self.algorithm.train_step(batch)
            
            self._total_steps += 1
            
            if step % 100 == 0:
                metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
                print(f"  Step {step}/{num_steps}: {metrics_str}")
    
    def save_checkpoint(self, path: str):
        """保存 checkpoint"""
        self.algorithm.model_group.save(path)
        print(f"[RECAP] Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """加载 checkpoint"""
        self.algorithm.model_group.load(path)
        print(f"[RECAP] Loaded checkpoint from {path}")
