"""
Value Function Model for RECAP (π0.6*)

Paper: Physical Intelligence π0.6* (Robotic Embodied CAlibratedPolicy)
https://www.physicalintelligence.company/download/recap.pdf

Value Function 架构:
- Vision Encoder: SIGLIP (400M) - 与 PaliGemma 不同
- Language Model: Gemma 3 (270M)
- Output: 201 bins for discretized value
- Loss: Cross-Entropy on bins

关键设计:
1. Value 离散化为 201 个 bins
2. 使用 C_neg = -1000 作为失败的惩罚
3. N-step lookahead: 50 steps
4. Normalization: 对 reward 进行 normalize
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .configuration import ValueConfig, RewardConfig


class SIGLIPVisionEncoder(nn.Module):
    """
    SIGLIP Vision Encoder
    
    比 PaliGemma 中的 SigLIP 更大的版本
    用于 Value Function (没有动作预测的额外需求)
    
    TODO: 使用实际的 HuggingFace SIGLIP model
    """
    
    def __init__(self, config: ValueConfig):
        super().__init__()
        self.config = config
        
        # Placeholder: 使用简单的 CNN 作为占位符
        # 实际应该使用 transformers 的 SiglipVisionModel
        self.patch_size = 14
        
        # Vision encoder (placeholder)
        self.patch_embed = nn.Conv2d(3, config.vision_dim, kernel_size=14, stride=14)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.vision_dim))
        
        # Positional embedding
        num_patches = (config.image_size // self.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.vision_dim))
        
        # Transformer layers (placeholder)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.vision_dim,
                nhead=config.vision_dim // 64,
                dim_feedforward=config.vision_dim * 4,
                dropout=config.dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(config.vision_depth)
        ])
        
        self.norm = nn.LayerNorm(config.vision_dim)
    
    def forward(self, images: Tensor) -> Tensor:
        """
        Args:
            images: (B, C, H, W) 或 (B, T, C, H, W) 图像
            
        Returns:
            (B, num_patches+1, vision_dim) 图像特征
        """
        if images.dim() == 5:
            B, T, C, H, W = images.shape
            images = images.view(B * T, C, H, W)
            batch_mode = "video"
        else:
            batch_mode = "image"
        
        # Patch embedding
        x = self.patch_embed(images)  # (B, dim, H/14, W/14)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, dim)
        
        # Add CLS token
        B_flat = x.shape[0]
        cls_tokens = self.cls_token.expand(B_flat, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        if batch_mode == "video":
            # Reshape back: (B*T, num_patches+1, dim) -> (B, T*(num_patches+1), dim)
            x = x.view(B, T * x.shape[1], x.shape[2])
        
        return x


class ValueTransformerDecoder(nn.Module):
    """
    Value Function 的 Transformer Decoder
    
    使用 Gemma 3 (270M) 作为 backbone
    TODO: 使用实际的 HuggingFace Gemma 3 model
    """
    
    def __init__(self, config: ValueConfig):
        super().__init__()
        self.config = config
        
        # Language embedding
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # Vision projection (对齐 vision_dim 到 hidden_dim)
        self.vision_proj = nn.Linear(config.vision_dim, config.hidden_dim)
        
        # Transformer decoder layers
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.ff_dim,
                dropout=config.dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(config.num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.hidden_dim)
        
        # Value head (输出 bins)
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.number_of_bins)
        )
    
    def forward(
        self,
        vision_features: Tensor,
        vision_mask: Tensor,
        lang_tokens: Tensor,
        lang_mask: Tensor,
    ) -> Tensor:
        """
        Args:
            vision_features: (B, num_patches, vision_dim)
            vision_mask: (B, num_patches)
            lang_tokens: (B, seq_len)
            lang_mask: (B, seq_len)
            
        Returns:
            (B, number_of_bins) value logits
        """
        # Project vision features
        vision_emb = self.vision_proj(vision_features)
        
        # Embed language tokens
        lang_emb = self.token_embed(lang_tokens)
        
        # Concatenate: [vision, language]
        seq_emb = torch.cat([vision_emb, lang_emb], dim=1)
        seq_mask = torch.cat([vision_mask, lang_mask], dim=1)
        
        # Create causal mask for decoder
        seq_len = seq_emb.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=seq_emb.device), diagonal=1
        ).bool()
        
        # Transformer decoder (using self-attention only for simplicity)
        x = seq_emb
        for layer in self.layers:
            x = layer(x, x, tgt_mask=causal_mask)
        
        x = self.norm(x)
        
        # Use last token for value prediction
        last_hidden = x[:, -1, :]
        value_logits = self.value_head(last_hidden)
        
        return value_logits


class ValueModel(nn.Module):
    """
    完整的 Value Function Model
    
    架构:
    ```
    ┌────────────────────────────────────────────┐
    │                                            │
    │   ┌───────────┐     ┌──────────────────┐   │
    │   │  SIGLIP   │────▶│ Gemma 3 Decoder  │   │
    │   │ (400M)    │     │ (270M)           │   │
    │   └───────────┘     └────────┬─────────┘   │
    │         ▲                    │             │
    │         │                    ▼             │
    │      images            ┌─────────┐        │
    │                        │Value Bins│        │
    │                        │ (201)   │        │
    │                        └─────────┘        │
    └────────────────────────────────────────────┘
    ```
    """
    
    def __init__(self, config: ValueConfig):
        super().__init__()
        self.config = config
        self.reward_config = config.reward_config
        
        self.vision_encoder = SIGLIPVisionEncoder(config)
        self.decoder = ValueTransformerDecoder(config)
        
        # 计算 bin edges
        self._setup_bins()
    
    def _setup_bins(self):
        """设置 value 离散化的 bins"""
        # Value range: [0, 1] (normalized)
        # 201 bins: 第一个 bin 对应 failure, 其余 200 bins 均匀分布
        
        self.register_buffer(
            "bin_edges",
            torch.linspace(0, 1, self.config.number_of_bins + 1)
        )
        self.register_buffer(
            "bin_centers",
            (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        )
    
    def value_to_bin(self, values: Tensor) -> Tensor:
        """
        将连续 value 转换为 bin index
        
        Args:
            values: (B,) normalized values in [0, 1]
            
        Returns:
            (B,) bin indices
        """
        # Clamp to [0, 1]
        values = values.clamp(0, 1)
        
        # Find bin index
        bin_idx = torch.bucketize(values, self.bin_edges[:-1]) - 1
        bin_idx = bin_idx.clamp(0, self.config.number_of_bins - 1)
        
        return bin_idx
    
    def bin_to_value(self, bin_idx: Tensor) -> Tensor:
        """
        将 bin index 转换为连续 value
        
        Args:
            bin_idx: (B,) bin indices
            
        Returns:
            (B,) normalized values
        """
        return self.bin_centers[bin_idx]
    
    def normalize_value(self, returns: Tensor) -> Tensor:
        """
        Normalize returns to [0, 1]
        
        公式: v_norm = (R - C_neg) / (reward_normalizer - C_neg)
        
        Args:
            returns: (B,) raw returns
            
        Returns:
            (B,) normalized values in [0, 1]
        """
        C_neg = self.reward_config.C_neg
        normalizer = self.reward_config.reward_normalizer
        
        v_norm = (returns - C_neg) / (normalizer - C_neg)
        v_norm = v_norm.clamp(0, 1)
        
        return v_norm
    
    def denormalize_value(self, v_norm: Tensor) -> Tensor:
        """
        Denormalize values from [0, 1] to original scale
        """
        C_neg = self.reward_config.C_neg
        normalizer = self.reward_config.reward_normalizer
        
        returns = v_norm * (normalizer - C_neg) + C_neg
        return returns
    
    def forward(
        self,
        images: Tensor,
        lang_tokens: Tensor,
        lang_mask: Tensor,
        returns: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        前向传播
        
        Args:
            images: (B, C, H, W) 或 (B, T, C, H, W)
            lang_tokens: (B, seq_len)
            lang_mask: (B, seq_len)
            returns: (B,) 可选，用于计算 loss
            
        Returns:
            包含 logits 和可选 loss 的字典
        """
        # Vision encoding
        vision_features = self.vision_encoder(images)
        vision_mask = torch.ones(vision_features.shape[:2], dtype=torch.bool, device=images.device)
        
        # Decoder
        value_logits = self.decoder(
            vision_features, vision_mask,
            lang_tokens, lang_mask
        )
        
        result = {"logits": value_logits}
        
        # Compute loss if targets provided
        if returns is not None:
            # Normalize and convert to bins
            v_norm = self.normalize_value(returns)
            target_bins = self.value_to_bin(v_norm)
            
            loss = F.cross_entropy(value_logits, target_bins)
            result["loss"] = loss
            result["target_bins"] = target_bins
        
        return result
    
    def predict_value(
        self,
        images: Tensor,
        lang_tokens: Tensor,
        lang_mask: Tensor,
    ) -> Tensor:
        """
        预测 value (推理)
        
        Args:
            images: (B, C, H, W)
            lang_tokens: (B, seq_len)
            lang_mask: (B, seq_len)
            
        Returns:
            (B,) predicted values (denormalized)
        """
        with torch.no_grad():
            output = self.forward(images, lang_tokens, lang_mask)
            
            # Softmax and expected value
            probs = F.softmax(output["logits"], dim=-1)
            v_norm = (probs * self.bin_centers.unsqueeze(0)).sum(dim=-1)
            
            # Denormalize
            values = self.denormalize_value(v_norm)
        
        return values


class ValueFunction(nn.Module):
    """
    Value Function Wrapper
    
    包装 ValueModel 并提供训练/推理接口
    """
    
    def __init__(self, config: ValueConfig):
        super().__init__()
        self.config = config
        self.model = ValueModel(config)
        
        # Language tokenizer (placeholder)
        self.tokenizer = None
    
    def compute_returns(
        self,
        rewards: Tensor,
        dones: Tensor,
        gamma: float = 0.99,
        n_steps: Optional[int] = None,
    ) -> Tensor:
        """
        计算 n-step returns
        
        Args:
            rewards: (B, T) rewards
            dones: (B, T) done flags
            gamma: discount factor
            n_steps: lookahead steps (default: from config)
            
        Returns:
            (B, T) n-step returns
        """
        if n_steps is None:
            n_steps = self.config.reward_config.N_steps_look_ahead
        
        B, T = rewards.shape
        returns = torch.zeros_like(rewards)
        
        for t in range(T):
            # n-step return from position t
            ret = 0
            for k in range(min(n_steps, T - t)):
                discount = gamma ** k
                ret += discount * rewards[:, t + k]
                
                # Stop if done
                if k < T - t - 1 and dones[:, t + k].any():
                    break
            
            returns[:, t] = ret
        
        return returns
    
    def compute_advantage(
        self,
        images: Tensor,
        lang_tokens: Tensor,
        lang_mask: Tensor,
        returns: Tensor,
    ) -> Tensor:
        """
        计算 advantage
        
        A(s) = R(s) - V(s)
        
        Args:
            images: (B, C, H, W)
            lang_tokens: (B, seq_len)
            lang_mask: (B, seq_len)
            returns: (B,) actual returns
            
        Returns:
            (B,) advantages
        """
        predicted_values = self.model.predict_value(images, lang_tokens, lang_mask)
        advantages = returns - predicted_values
        
        return advantages
    
    def forward(
        self,
        images: Tensor,
        lang_tokens: Tensor,
        lang_mask: Tensor,
        returns: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """前向传播"""
        return self.model.forward(images, lang_tokens, lang_mask, returns)
    
    def predict(
        self,
        images: Tensor,
        lang_tokens: Tensor,
        lang_mask: Tensor,
    ) -> Tensor:
        """预测 value"""
        return self.model.predict_value(images, lang_tokens, lang_mask)


def compute_n_step_returns(
    rewards: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    n_steps: int = 50,
) -> np.ndarray:
    """
    Numpy 版本的 n-step return 计算
    
    用于数据预处理
    
    Args:
        rewards: (T,) rewards
        dones: (T,) done flags
        gamma: discount factor
        n_steps: lookahead steps
        
    Returns:
        (T,) n-step returns
    """
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    
    for t in range(T):
        ret = 0
        for k in range(min(n_steps, T - t)):
            discount = gamma ** k
            ret += discount * rewards[t + k]
            
            if dones[t + k]:
                break
        
        returns[t] = ret
    
    return returns


def compute_advantage_indicator(
    advantages: np.ndarray,
    threshold: float = 0.0,
) -> np.ndarray:
    """
    计算 advantage indicator I_t
    
    I_t = 1 if A(s_t) > threshold else 0
    
    用于 AWR 中筛选高 advantage 样本
    
    Args:
        advantages: (T,) advantages
        threshold: advantage 阈值
        
    Returns:
        (T,) binary indicators
    """
    return (advantages > threshold).astype(np.float32)
