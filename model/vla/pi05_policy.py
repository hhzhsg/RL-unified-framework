"""
π0.5 Policy 实现

π0.5: A Vision-Language-Action Flow Model with Discrete Actions
Paper: https://www.physicalintelligence.company/download/pi05.pdf

相比 π0 的改进:
- 离散动作 tokenization (FAST tokenizer)
- Knowledge Insulation (KI) - 阻断 VLM 到 Expert 的梯度
- Adaptive RMS Norm (AdaRMS)
- 更长的 tokenizer 上下文
"""
import math
from collections import deque
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.base_policy import BasePolicy
from data import Observation, RobotState, Action
from .configuration import PI05Config
from .paligemma_with_expert import PaliGemmaWithExpertConfig, PaliGemmaWithExpertModel
from .pi0_policy import fourier_embed, make_att_2d_masks


class PI05FlowMatching(nn.Module):
    """
    π0.5 Flow Matching Model
    
    在 π0 基础上增加:
    - 离散动作 token 预测 (Cross-Entropy loss)
    - Knowledge Insulation
    - AdaRMS conditioning
    
    架构:
    ```
    ┌──────────────────────────────────────────┐
    │                   actions                │
    │                   ▲                      │
    │                  ┌┴─────┐                │
    │      kv cache    │Gemma │                │
    │      ┌──────────►│Expert│◄─ AdaRMS cond  │
    │      │           │      │                │
    │     ┌┴─────────┐ │x 10  │                │
    │     │          │ └▲─────┘                │
    │     │PaliGemma │  │                      │
    │     │          │  noise                  │
    │     └▲──▲──▲──▲                          │
    │      │  │  │  └── discrete actions       │
    │      │  │  language                      │
    │      │  state                            │
    │      image(s)                            │
    └──────────────────────────────────────────┘
    ```
    """
    
    def __init__(
        self,
        config: PI05Config,
        discrete_action_vocab_size: Optional[int] = None,
        paligemma_config: PaliGemmaWithExpertConfig = None,
        use_tiny: bool = False,
    ):
        super().__init__()
        self.config = config
        
        # 使用配置中的或传入的 vocab size
        vocab_size = discrete_action_vocab_size or config.discrete_action_vocab_size
        
        # 如果使用 tiny 配置
        if use_tiny:
            paligemma_config = PaliGemmaWithExpertConfig.tiny()
            paligemma_config.discrete_action_vocab_size = vocab_size
            paligemma_config.use_adarms = config.use_adarms
            paligemma_config.adarms_cond_dim = config.proj_width
        elif paligemma_config is None:
            # 初始化 PaliGemma with Expert
            paligemma_config = PaliGemmaWithExpertConfig(
                freeze_vision_encoder=config.freeze_vision_encoder,
                train_expert_only=config.train_expert_only,
                attention_implementation=config.attention_implementation,
                discrete_action_vocab_size=vocab_size,
                dropout=config.dropout,
                use_adarms=config.use_adarms,
                adarms_cond_dim=config.proj_width,
            )
        
        self.paligemma_with_expert = PaliGemmaWithExpertModel(paligemma_config)
        
        # 保存 hidden sizes
        self.vlm_hidden_size = paligemma_config.paligemma_hidden_size
        self.expert_hidden_size = paligemma_config.expert_hidden_size
        
        # Action projections
        self.action_in_proj = nn.Linear(config.max_action_dim, config.proj_width)
        self.action_out_proj = nn.Linear(config.proj_width, config.max_action_dim)
        
        # Suffix to VLM projection
        self.suffix_proj = nn.Linear(config.proj_width, self.vlm_hidden_size)
        
        # Expert output projection
        self.expert_out_proj = nn.Linear(self.expert_hidden_size, config.proj_width)
        
        # Time MLP (用于 AdaRMS conditioning)
        self.time_mlp_in = nn.Linear(config.proj_width, config.proj_width)
        self.time_mlp_out = nn.Linear(config.proj_width, config.proj_width)
        
        # 初始化权重
        self._init_model()
    
    def _init_weights(self, module: nn.Module):
        """He 初始化"""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='linear')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def _init_model(self):
        """根据配置初始化模型"""
        if self.config.init_strategy == "no_init":
            return
        elif self.config.init_strategy == "full_he_init":
            for m in self.modules():
                self._init_weights(m)
        elif self.config.init_strategy == "expert_only_he_init":
            for m in self.paligemma_with_expert.expert_layers.modules():
                self._init_weights(m)
    
    def sample_noise(self, shape: Tuple[int, ...], device: torch.device) -> Tensor:
        """采样噪声"""
        return torch.randn(shape, device=device)
    
    def sample_time(self, bsize: int, device: torch.device) -> Tensor:
        """采样时间步"""
        return torch.rand(bsize, device=device)
    
    def embed_prefix(
        self,
        images: List[Tensor],
        img_masks: List[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        discrete_actions: Optional[Tensor] = None,
        discrete_action_masks: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        编码 prefix (图像 + 语言 + 可选的离散动作)
        
        Args:
            images: 图像列表
            img_masks: 图像 mask
            lang_tokens: 语言 tokens
            lang_masks: 语言 mask
            discrete_actions: (B, seq_len) 离散动作 tokens (可选)
            discrete_action_masks: (B, seq_len) 离散动作 mask
            
        Returns:
            embs, pad_masks, att_masks
        """
        embs = []
        pad_masks = []
        att_masks = []
        
        # 编码图像
        for img, img_mask in zip(images, img_masks):
            img_emb = self.paligemma_with_expert.embed_image(img)
            # 不再手动转换 dtype
            embs.append(img_emb)
            pad_masks.append(img_mask)
            
            num_img_tokens = img_emb.shape[1]
            att_masks += [0] * num_img_tokens
        
        # 编码语言
        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        # 不再手动转换 dtype
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        
        num_lang_tokens = lang_emb.shape[1]
        att_masks += [0] * num_lang_tokens
        
        # 编码离散动作 (如果有)
        if discrete_actions is not None:
            da_emb = self.paligemma_with_expert.embed_discrete_actions(discrete_actions)
            # 不再手动转换 dtype
            embs.append(da_emb)
            pad_masks.append(discrete_action_masks)
            
            # 离散动作使用 causal attention
            att_masks += [1] * da_emb.shape[1]
        
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        
        return embs, pad_masks, att_masks
    
    def embed_suffix(
        self,
        noisy_actions: Tensor,
        timestep: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        编码 suffix (noisy actions + time)
        
        π0.5 不使用 robot state，只使用离散化的 state 在 language prompt 中
        
        Args:
            noisy_actions: (B, n_action_steps, action_dim) 带噪声的动作
            timestep: (B,) 时间步
            
        Returns:
            embs, pad_masks, att_masks, adarms_cond
        """
        embs = []
        pad_masks = []
        att_masks = []
        
        bsize = noisy_actions.shape[0]
        device = noisy_actions.device
        dtype = noisy_actions.dtype
        
        # Time embedding
        time_emb = fourier_embed(timestep, self.config.proj_width, device=device)
        
        # Time MLP for AdaRMS conditioning
        def time_mlp_func(x):
            x = self.time_mlp_in(x)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return x
        
        time_emb = time_emb.to(dtype=dtype)
        adarms_cond = time_mlp_func(time_emb)
        
        # Action embedding
        action_emb = self.action_in_proj(noisy_actions)
        embs.append(action_emb)
        
        action_mask = torch.ones(bsize, action_emb.shape[1], dtype=torch.bool, device=device)
        pad_masks.append(action_mask)
        
        # Causal attention for actions
        att_masks += [1] + ([0] * (self.config.n_action_steps - 1))
        
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        
        return embs, pad_masks, att_masks, adarms_cond
    
    def forward(
        self,
        images: List[Tensor],
        img_masks: List[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        actions: Tensor,
        noise: Optional[Tensor] = None,
        time: Optional[Tensor] = None,
        discrete_actions: Optional[Tensor] = None,
        discrete_action_masks: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        训练前向传播
        
        Args:
            images: 图像列表
            img_masks: 图像 mask
            lang_tokens: 语言 tokens
            lang_masks: 语言 mask
            actions: 目标连续动作
            noise: 可选噪声
            time: 可选时间步
            discrete_actions: 目标离散动作 tokens
            discrete_action_masks: 离散动作 mask
            
        Returns:
            包含 MSE 和 CE loss 的字典
        """
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        
        # 计算 noisy actions
        time_expanded = time[:, None, None]
        noisy_actions = time_expanded * noise + (1 - time_expanded) * actions
        
        # 目标 velocity
        u_t = noise - actions
        
        # Embed prefix (with KV cache)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks,
            discrete_actions, discrete_action_masks
        )
        
        # Build VLM attention mask
        vlm_2d_attention_mask = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        vlm_position_ids = torch.cumsum(prefix_pad_masks.long(), dim=1) - 1
        
        num_cross_att_tokens = prefix_embs.shape[1] - self.config.discrete_action_max_length if discrete_actions is not None else prefix_embs.shape[1]
        
        # Forward through VLM (cache KV)
        (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=vlm_2d_attention_mask,
            position_ids=vlm_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            n_cross_att_tokens=num_cross_att_tokens,
            use_cache=True,
            fill_kv_cache=True,
        )
        
        # Embed suffix
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            noisy_actions, time
        )
        
        # Build full attention mask for expert
        total_len = prefix_pad_masks.shape[1] + suffix_pad_masks.shape[1]
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = prefix_att_masks + suffix_att_masks
        
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        
        # Compute position ids for expert (skip discrete action tokens)
        if discrete_actions is not None:
            prefix_offsets = torch.sum(prefix_pad_masks[:, :-self.config.discrete_action_max_length], dim=-1)[:, None]
        else:
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        
        action_expert_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks.long(), dim=1) - 1
        
        # Knowledge Insulation: stop gradient from expert to VLM
        if self.config.use_knowledge_insulation and past_key_values is not None:
            for layer_kv in past_key_values:
                if layer_kv is not None:
                    for key in layer_kv:
                        layer_kv[key] = layer_kv[key].detach()
        
        # Forward through expert
        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=action_expert_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=True,
            fill_kv_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        
        # Compute velocity prediction (MSE loss)
        suffix_out = suffix_out[:, -self.config.n_action_steps:]
        v_t = self.action_out_proj(suffix_out)
        v_t = v_t.to(dtype=torch.float32)
        
        mse_loss = F.mse_loss(v_t, u_t, reduction="none")
        mse_loss = mse_loss.mean()
        
        # Compute discrete action loss (CE)
        ce_loss = torch.zeros_like(mse_loss)
        if discrete_actions is not None and hasattr(self.paligemma_with_expert, 'da_head'):
            # 从 prefix_out 获取离散动作预测
            da_logits = self.paligemma_with_expert.da_head(prefix_out[:, -self.config.discrete_action_max_length:])
            
            batch_size, seq_len = discrete_actions.shape
            da_logits = da_logits.reshape(batch_size * seq_len, -1)
            da_targets = discrete_actions.reshape(batch_size * seq_len)
            
            ce_loss = F.cross_entropy(da_logits, da_targets.long(), reduction='none')
            ce_loss = ce_loss.reshape(batch_size, seq_len)
            
            # Mask padding
            if discrete_action_masks is not None:
                ce_loss = ce_loss * discrete_action_masks.float()
            
            ce_loss = ce_loss.mean()
        
        return {"MSE": mse_loss, "CE": ce_loss}
    
    def sample_actions(
        self,
        images: List[Tensor],
        img_masks: List[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        """
        采样动作 (推理)
        
        Args:
            images: 图像列表
            img_masks: 图像 mask
            lang_tokens: 语言 tokens
            lang_masks: 语言 mask
            noise: 可选初始噪声
            
        Returns:
            (B, n_action_steps, action_dim) 采样的动作
        """
        bsize = lang_tokens.shape[0]
        device = lang_tokens.device
        
        if noise is None:
            actions_shape = (bsize, self.config.n_action_steps, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)
        
        # Embed prefix and cache KV
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks.long(), dim=1) - 1
        
        num_cross_att_tokens = prefix_embs.shape[1]
        
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            n_cross_att_tokens=num_cross_att_tokens,
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        
        # Euler integration
        dt = -1.0 / self.config.num_steps
        x_t = noise
        
        for step in range(self.config.num_steps):
            t = 1.0 - step / self.config.num_steps
            timestep = torch.full((bsize,), t, device=device)
            
            x_t = self._denoise_step(
                prefix_pad_masks, past_key_values,
                x_t, timestep, prefix_att_masks
            )
            
            if step < self.config.num_steps - 1:
                # Predict velocity and step
                pass  # velocity is computed inside _denoise_step
        
        return x_t
    
    def _denoise_step(
        self,
        prefix_pad_masks: Tensor,
        past_key_values: List[Dict[str, Tensor]],
        x_t: Tensor,
        timestep: Tensor,
        prefix_att_masks: List[int],
    ) -> Tensor:
        """
        单步 denoising
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            x_t, timestep
        )
        
        # Build attention mask
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = prefix_att_masks + suffix_att_masks
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        action_expert_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks.long(), dim=1) - 1
        
        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=action_expert_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=True,
            fill_kv_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        
        suffix_out = suffix_out[:, -self.config.n_action_steps:]
        v_t = self.action_out_proj(suffix_out)
        v_t = v_t.to(dtype=torch.float32)
        
        # Euler step
        dt = -1.0 / self.config.num_steps
        x_t = x_t + v_t * dt
        
        return x_t


class PI05Policy(BasePolicy):
    """
    π0.5 Policy Wrapper
    """
    
    def __init__(
        self,
        config: PI05Config,
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        use_tiny: bool = False,
    ):
        state_dim = state_dim or config.max_state_dim
        action_dim = action_dim or config.max_action_dim
        
        super().__init__(state_dim, action_dim)
        
        self.config = config
        
        # TODO: 从 FAST processor 获取 vocab size
        discrete_action_vocab_size = config.discrete_action_vocab_size or 262144  # FAST default
        
        self.model = PI05FlowMatching(config, discrete_action_vocab_size=discrete_action_vocab_size, use_tiny=use_tiny)
        
        # Action queue
        self._action_queue = deque([], maxlen=config.n_action_steps)
        
        # Tokenizers (placeholder)
        self.language_tokenizer = None
        self.discrete_action_processor = None
    
    def reset(self):
        """重置 action queue"""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)
    
    def forward(
        self,
        obs: Dict[str, Tensor],
        robot_state: Tensor,
    ) -> Tensor:
        """前向传播 (推理)"""
        images, img_masks = self._prepare_images(obs)
        lang_tokens, lang_masks = self._prepare_language(obs)
        
        actions = self.model.sample_actions(
            images, img_masks,
            lang_tokens, lang_masks,
        )
        
        return actions
    
    def compute_loss(
        self,
        obs: Dict[str, Tensor],
        robot_state: Tensor,
        actions: Tensor,
        discrete_actions: Optional[Tensor] = None,
        discrete_action_masks: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """计算训练 loss"""
        images, img_masks = self._prepare_images(obs)
        lang_tokens, lang_masks = self._prepare_language(obs)
        
        losses = self.model.forward(
            images, img_masks,
            lang_tokens, lang_masks,
            actions,
            discrete_actions=discrete_actions,
            discrete_action_masks=discrete_action_masks,
        )
        
        return losses
    
    def act(
        self,
        obs: Observation,
        robot_state: RobotState,
        deterministic: bool = True,
    ) -> Action:
        """推理接口"""
        self.eval()
        
        if len(self._action_queue) == 0:
            with torch.no_grad():
                obs_dict = self._obs_to_dict(obs)
                state_tensor = self._state_to_tensor(robot_state)
                
                actions = self.forward(obs_dict, state_tensor)
                
                for i in range(actions.shape[1]):
                    self._action_queue.append(actions[0, i].cpu().numpy())
        
        action_data = self._action_queue.popleft()
        return Action(data=action_data, space="joint")
    
    def _prepare_images(self, obs: Dict[str, Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """准备图像输入"""
        images = []
        masks = []
        
        for key in self.config.image_features:
            if key in obs:
                img = obs[key]
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                images.append(img)
                
                num_patches = (self.config.resize_imgs_with_padding[0] // 14) * (self.config.resize_imgs_with_padding[1] // 14)
                masks.append(torch.ones(img.shape[0], num_patches, dtype=torch.bool, device=img.device))
        
        return images, masks
    
    def _prepare_language(self, obs: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        """准备语言输入"""
        device = next(self.parameters()).device
        
        if "language_tokens" in obs:
            return obs["language_tokens"], obs["language_masks"]
        
        batch_size = 1
        tokens = torch.zeros(batch_size, self.config.tokenizer_max_length, dtype=torch.long, device=device)
        masks = torch.ones(batch_size, self.config.tokenizer_max_length, dtype=torch.bool, device=device)
        
        return tokens, masks
    
    def _obs_to_dict(self, obs: Observation) -> Dict[str, Tensor]:
        """将 Observation 转为字典"""
        device = next(self.parameters()).device
        result = {}
        
        for name, img in obs.images.items():
            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
            result[f"observation.{name}"] = img_tensor.to(device)
        
        result["language"] = obs.language
        
        return result
    
    def _state_to_tensor(self, state: RobotState) -> Tensor:
        """将 RobotState 转为 Tensor"""
        device = next(self.parameters()).device
        state_array = state.to_array()
        
        if len(state_array) < self.config.max_state_dim:
            padded = np.zeros(self.config.max_state_dim, dtype=np.float32)
            padded[:len(state_array)] = state_array
            state_array = padded
        
        return torch.from_numpy(state_array).float().unsqueeze(0).to(device)
    
    def get_optim_params(self):
        """获取需要优化的参数"""
        return self.parameters()
