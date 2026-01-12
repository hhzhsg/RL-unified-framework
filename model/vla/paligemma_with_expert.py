"""
PaliGemma with Expert Model

将 PaliGemma VLM 与 Gemma Expert 模型结合用于动作生成

这是 π0 和 π0.5 的核心模型组件:
- PaliGemma: Vision-Language Model (视觉+语言编码)
- Gemma Expert: Action Decoder (动作解码)
- 两者共享 KV Cache 进行高效推理
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class PaliGemmaWithExpertConfig:
    """
    PaliGemma + Expert 配置
    """
    # Vision encoder
    vision_hidden_size: int = 1152
    vision_intermediate_size: int = 4304
    vision_num_attention_heads: int = 16
    vision_num_hidden_layers: int = 27
    num_image_tokens: int = 256
    patch_size: int = 14
    
    # PaliGemma language model
    paligemma_hidden_size: int = 2048
    paligemma_intermediate_size: int = 16384
    paligemma_num_attention_heads: int = 8
    paligemma_num_hidden_layers: int = 18
    paligemma_num_key_value_heads: int = 1
    paligemma_vocab_size: int = 257152
    
    # Gemma Expert (Action decoder)
    expert_hidden_size: int = 1024
    expert_intermediate_size: int = 4096
    expert_num_attention_heads: int = 8
    expert_num_hidden_layers: int = 18
    expert_num_key_value_heads: int = 1
    
    # AdaRMS (Adaptive RMS Norm) - π0.5 特性
    use_adarms: bool = False
    adarms_cond_dim: int = 1024
    
    # 离散动作 (π0.5)
    discrete_action_vocab_size: Optional[int] = None
    
    # 训练配置
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    attention_implementation: str = "eager"
    load_pretrained_paligemma: bool = False
    dropout: float = 0.1
    
    @classmethod
    def tiny(cls):
        """创建一个用于测试的小配置"""
        return cls(
            vision_hidden_size=256,
            vision_intermediate_size=512,
            vision_num_attention_heads=4,
            vision_num_hidden_layers=2,
            num_image_tokens=64,
            paligemma_hidden_size=256,
            paligemma_intermediate_size=512,
            paligemma_num_attention_heads=4,
            paligemma_num_hidden_layers=2,
            paligemma_num_key_value_heads=1,
            expert_hidden_size=256,
            expert_intermediate_size=512,
            expert_num_attention_heads=4,
            expert_num_hidden_layers=2,
            expert_num_key_value_heads=1,
            adarms_cond_dim=256,
        )


class PaliGemmaWithExpertModel(nn.Module):
    """
    PaliGemma with Expert Model
    
    架构:
    ```
    ┌────────────────────────────────────┐
    │           actions                   │
    │              ▲                      │
    │         ┌────┴────┐                │
    │  kv     │  Gemma  │                │
    │  cache  │  Expert │ ◄─ noise/time  │
    │    ┌───►│         │                │
    │    │    └─────────┘                │
    │ ┌──┴──────────┐                    │
    │ │  PaliGemma  │                    │
    │ │   (VLM)     │                    │
    │ └──▲──────▲───┘                    │
    │    │      │                        │
    │  images  language                  │
    └────────────────────────────────────┘
    ```
    """
    
    def __init__(self, config: PaliGemmaWithExpertConfig):
        super().__init__()
        self.config = config
        
        # 简化实现: 使用 placeholder 层
        # 实际部署时替换为 HuggingFace 模型
        
        # Vision encoder (SigLIP)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, config.vision_hidden_size, kernel_size=config.patch_size, stride=config.patch_size),
            nn.GELU(),
            nn.Flatten(2),
        )
        self.vision_proj = nn.Linear(config.vision_hidden_size, config.paligemma_hidden_size)
        
        # Language embedding
        self.embed_tokens = nn.Embedding(config.paligemma_vocab_size, config.paligemma_hidden_size)
        
        # PaliGemma layers (simplified)
        self.paligemma_layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=config.paligemma_hidden_size,
                num_heads=config.paligemma_num_attention_heads,
                intermediate_size=config.paligemma_intermediate_size,
                dropout=config.dropout,
            )
            for _ in range(config.paligemma_num_hidden_layers)
        ])
        
        # Gemma Expert layers
        self.expert_layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=config.expert_hidden_size,
                num_heads=config.expert_num_attention_heads,
                intermediate_size=config.expert_intermediate_size,
                dropout=config.dropout,
                use_adarms=config.use_adarms,
                adarms_cond_dim=config.adarms_cond_dim,
            )
            for _ in range(config.expert_num_hidden_layers)
        ])
        
        # Expert input projection (from PaliGemma dim to Expert dim)
        self.expert_in_proj = nn.Linear(config.paligemma_hidden_size, config.expert_hidden_size)
        
        # Discrete action embedding (for π0.5)
        if config.discrete_action_vocab_size is not None:
            self.discrete_action_embedding = nn.Embedding(
                config.discrete_action_vocab_size,
                config.paligemma_hidden_size,
                padding_idx=0,
            )
            self.da_head = nn.Linear(
                config.paligemma_hidden_size,
                config.discrete_action_vocab_size,
            )
        
        self.dropout = nn.Dropout(config.dropout)
        
        self._set_requires_grad()
    
    def _set_requires_grad(self):
        """设置哪些参数需要梯度"""
        if self.config.freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            for param in self.vision_proj.parameters():
                param.requires_grad = False
        
        if self.config.train_expert_only:
            for param in self.paligemma_layers.parameters():
                param.requires_grad = False
            for param in self.embed_tokens.parameters():
                param.requires_grad = False
    
    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        编码图像
        
        Args:
            image: (B, C, H, W) 图像张量
            
        Returns:
            (B, num_patches, hidden_size) 图像嵌入
        """
        # image: (B, C, H, W) -> (B, hidden, H/patch, W/patch)
        x = self.vision_encoder[0](image)
        x = self.vision_encoder[1](x)  # GELU
        # (B, hidden, H/patch, W/patch) -> (B, hidden, num_patches)
        x = x.flatten(2)
        # (B, hidden, num_patches) -> (B, num_patches, hidden)
        x = x.transpose(1, 2)
        # Project to PaliGemma dimension
        x = self.vision_proj(x)
        return x
    
    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        编码语言 tokens
        
        Args:
            tokens: (B, seq_len) token ids
            
        Returns:
            (B, seq_len, hidden_size) 语言嵌入
        """
        return self.embed_tokens(tokens)
    
    def embed_discrete_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        编码离散动作 (π0.5)
        
        Args:
            actions: (B, seq_len) 离散动作 token ids
            
        Returns:
            (B, seq_len, hidden_size) 动作嵌入
        """
        if not hasattr(self, 'discrete_action_embedding'):
            raise ValueError("Discrete action embedding not initialized. Set discrete_action_vocab_size in config.")
        return self.discrete_action_embedding(actions.long())
    
    def forward(
        self,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Optional[List[Dict[str, torch.Tensor]]] = None,
        inputs_embeds: List[torch.Tensor] = None,
        n_cross_att_tokens: Optional[int] = None,
        use_cache: bool = False,
        fill_kv_cache: bool = False,
        adarms_cond: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[List[Optional[torch.Tensor]], Optional[List[Dict[str, torch.Tensor]]]]:
        """
        前向传播
        
        Args:
            attention_mask: (B, seq_len, seq_len) 2D 注意力掩码
            position_ids: (B, seq_len) 位置 ids
            past_key_values: 缓存的 KV
            inputs_embeds: [prefix_embs, suffix_embs] 输入嵌入列表
            n_cross_att_tokens: 交叉注意力 token 数
            use_cache: 是否使用 KV cache
            fill_kv_cache: 是否填充 KV cache
            adarms_cond: AdaRMS 条件 (用于 π0.5)
            
        Returns:
            outputs_embeds: [prefix_out, suffix_out] 输出嵌入
            past_key_values: 更新的 KV cache
        """
        prefix_embs, suffix_embs = inputs_embeds if inputs_embeds else [None, None]
        
        if adarms_cond is None:
            adarms_cond = [None, None]
        
        new_past_key_values = [] if use_cache else None
        
        # 处理 prefix (PaliGemma)
        prefix_out = None
        if prefix_embs is not None:
            hidden_states = prefix_embs
            prefix_len = prefix_embs.shape[1]
            
            # 为 prefix 创建 attention mask (full attention for VLM)
            prefix_attention_mask = None  # 默认 full attention
            prefix_position_ids = position_ids[:, :prefix_len] if position_ids is not None else None
            
            for i, layer in enumerate(self.paligemma_layers):
                layer_past = past_key_values[i] if past_key_values else None
                hidden_states, new_kv = layer(
                    hidden_states,
                    attention_mask=prefix_attention_mask,
                    position_ids=prefix_position_ids,
                    past_key_values=layer_past,
                    use_cache=use_cache,
                )
                if use_cache:
                    new_past_key_values.append(new_kv)
            prefix_out = hidden_states
        
        # 处理 suffix (Expert)
        suffix_out = None
        if suffix_embs is not None:
            # Project to expert dimension
            hidden_states = self.expert_in_proj(suffix_embs) if suffix_embs.shape[-1] != self.config.expert_hidden_size else suffix_embs
            
            # 对于 expert layers，创建简单的 causal mask
            suffix_len = suffix_embs.shape[1]
            suffix_attention_mask = None  # 使用默认的 causal attention
            suffix_position_ids = position_ids[:, -suffix_len:] if position_ids is not None else None
            
            for i, layer in enumerate(self.expert_layers):
                # 从 prefix 的 KV cache 获取交叉注意力
                layer_past = new_past_key_values[i] if new_past_key_values and i < len(new_past_key_values) else None
                
                hidden_states, _ = layer(
                    hidden_states,
                    attention_mask=suffix_attention_mask,
                    position_ids=suffix_position_ids,
                    past_key_values=layer_past,
                    use_cache=False,
                    adarms_cond=adarms_cond[1],
                )
            suffix_out = hidden_states
        
        return [prefix_out, suffix_out], new_past_key_values if use_cache else None
    
    def to_bfloat16(self):
        """转换为 bfloat16 精度"""
        self.paligemma_layers = self.paligemma_layers.to(torch.bfloat16)
        self.expert_layers = self.expert_layers.to(torch.bfloat16)
        return self


class TransformerBlock(nn.Module):
    """
    Transformer Block (简化版)
    
    支持:
    - Multi-head Self-Attention
    - Optional AdaRMS (Adaptive RMS Norm)
    - KV Cache
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float = 0.1,
        use_adarms: bool = False,
        adarms_cond_dim: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Self-attention
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        # FFN
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        
        # Norms
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=1e-6)
        
        # AdaRMS (optional)
        self.use_adarms = use_adarms
        if use_adarms and adarms_cond_dim:
            self.adarms_scale = nn.Linear(adarms_cond_dim, hidden_size)
            self.adarms_shift = nn.Linear(adarms_cond_dim, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Dict[str, torch.Tensor]] = None,
        use_cache: bool = False,
        adarms_cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        前向传播
        """
        residual = hidden_states
        
        # Apply AdaRMS if enabled
        if self.use_adarms and adarms_cond is not None:
            scale = self.adarms_scale(adarms_cond).unsqueeze(1)
            shift = self.adarms_shift(adarms_cond).unsqueeze(1)
            hidden_states = self.input_layernorm(hidden_states) * (1 + scale) + shift
        else:
            hidden_states = self.input_layernorm(hidden_states)
        
        # Self-attention
        bsz, seq_len, _ = hidden_states.shape
        
        query = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Concat with past KV if exists
        new_kv = None
        if past_key_values is not None:
            key = torch.cat([past_key_values['key_states'], key], dim=2)
            value = torch.cat([past_key_values['value_states'], value], dim=2)
        
        if use_cache:
            new_kv = {'key_states': key, 'value_states': value}
        
        # Attention
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask.unsqueeze(1)
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        attn_output = self.dropout(attn_output)
        
        # Residual
        hidden_states = residual + attn_output
        
        # FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        hidden_states = self.down_proj(gate * up)
        hidden_states = self.dropout(hidden_states)
        
        hidden_states = residual + hidden_states
        
        return hidden_states, new_kv
