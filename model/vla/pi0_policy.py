"""
π0 Policy 实现

π0: A Vision-Language-Action Flow Model for General Robot Control
Paper: https://www.physicalintelligence.company/download/pi0.pdf

核心特点:
- Flow Matching 用于连续动作生成
- PaliGemma (3B) 作为 VLM backbone
- Gemma Expert 作为 action decoder
- 支持多相机输入
"""
import math
from collections import deque
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.base_policy import BasePolicy
from data import Observation, RobotState, Action
from .configuration import PI0Config
from .paligemma_with_expert import PaliGemmaWithExpertConfig, PaliGemmaWithExpertModel


def fourier_embed(x: Tensor, dim: int, min_period: float = 4e-3, max_period: float = 4.0, device=None) -> Tensor:
    """
    Fourier embedding for timestep
    
    Args:
        x: (B,) 时间步
        dim: embedding 维度
        min_period: 最小周期
        max_period: 最大周期
        
    Returns:
        (B, dim) embedding
    """
    if device is None:
        device = x.device
    
    freqs = torch.exp(
        torch.linspace(math.log(min_period), math.log(max_period), dim // 2, device=device)
    )
    x = x[:, None] * freqs[None, :]
    x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
    return x


def make_att_2d_masks(pad_masks: Tensor, att_masks: Tensor) -> Tensor:
    """
    构建 2D 注意力掩码
    
    Args:
        pad_masks: (B, seq_len) padding mask (True = valid)
        att_masks: (seq_len,) 注意力 pattern
        
    Returns:
        (B, seq_len, seq_len) 2D attention mask
    """
    bsz, seq_len = pad_masks.shape
    
    # Causal mask
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=pad_masks.device, dtype=torch.bool),
        diagonal=1
    )
    
    # Combine with padding
    pad_mask_2d = pad_masks[:, None, :] & pad_masks[:, :, None]
    
    # 使用 att_masks 控制哪些位置可以被 attend
    if isinstance(att_masks, (list, tuple)):
        att_masks = torch.tensor(att_masks, device=pad_masks.device)
    
    # Final mask: (B, seq_len, seq_len)
    mask = pad_mask_2d & ~causal_mask
    
    # 转换为 float mask (-inf for masked positions)
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, 0.0)
    
    return mask


class PI0FlowMatching(nn.Module):
    """
    π0 Flow Matching Model
    
    使用 Flow Matching 进行连续动作生成
    
    架构:
    ```
    ┌──────────────────────────────┐
    │               actions        │
    │               ▲              │
    │              ┌┴─────┐        │
    │  kv cache    │Gemma │        │
    │  ┌──────────►│Expert│        │
    │  │           │      │        │
    │ ┌┴────────┐  │x 10  │        │
    │ │         │  └▲──▲──┘        │
    │ │PaliGemma│   │  │           │
    │ │         │   │  robot state │
    │ │         │   noise          │
    │ └▲──▲─────┘                  │
    │  │  │                        │
    │  image(s)                    │
    │  language tokens             │
    └──────────────────────────────┘
    ```
    """
    
    def __init__(self, config: PI0Config, paligemma_config: PaliGemmaWithExpertConfig = None):
        super().__init__()
        self.config = config
        
        # 初始化 PaliGemma with Expert
        if paligemma_config is None:
            paligemma_config = PaliGemmaWithExpertConfig(
                freeze_vision_encoder=config.freeze_vision_encoder,
                train_expert_only=config.train_expert_only,
                attention_implementation=config.attention_implementation,
                dropout=config.dropout,
            )
        self.paligemma_with_expert = PaliGemmaWithExpertModel(paligemma_config)
        
        # State projection
        self.state_proj = nn.Linear(config.max_state_dim, config.proj_width)
        
        # Action projections
        self.action_in_proj = nn.Linear(config.max_action_dim, config.proj_width)
        self.action_out_proj = nn.Linear(config.proj_width, config.max_action_dim)
        
        # Time + Action MLP
        self.action_time_mlp_in = nn.Linear(config.proj_width * 2, config.proj_width)
        self.action_time_mlp_out = nn.Linear(config.proj_width, config.proj_width)
        
        # Suffix to VLM projection (对齐 suffix 到 paligemma hidden size)
        self.suffix_proj = nn.Linear(config.proj_width, paligemma_config.paligemma_hidden_size)
        
        # Expert output projection (从 expert hidden 到 proj_width)
        self.expert_out_proj = nn.Linear(paligemma_config.expert_hidden_size, config.proj_width)
        
        # 保存 hidden sizes
        self.vlm_hidden_size = paligemma_config.paligemma_hidden_size
        self.expert_hidden_size = paligemma_config.expert_hidden_size
        
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
        """采样时间步 (uniform in [0, 1])"""
        return torch.rand(bsize, device=device)
    
    def embed_prefix(
        self,
        images: List[Tensor],
        img_masks: List[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        编码 prefix (图像 + 语言)
        
        Args:
            images: 图像列表 [(B, C, H, W), ...]
            img_masks: 图像 mask 列表
            lang_tokens: (B, seq_len) 语言 tokens
            lang_masks: (B, seq_len) 语言 mask
            
        Returns:
            embs: (B, total_len, hidden) 嵌入
            pad_masks: (B, total_len) padding mask
            att_masks: 注意力 pattern
        """
        embs = []
        pad_masks = []
        att_masks = []
        
        # 编码每个图像
        for img, img_mask in zip(images, img_masks):
            img_emb = self.paligemma_with_expert.embed_image(img)
            # 不再手动转换 dtype
            embs.append(img_emb)
            pad_masks.append(img_mask)
            
            # Full attention for images
            num_img_tokens = img_emb.shape[1]
            att_masks += [0] * num_img_tokens
        
        # 编码语言
        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        # 不再手动转换 dtype
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        
        # Full attention for language
        num_lang_tokens = lang_emb.shape[1]
        att_masks += [0] * num_lang_tokens
        
        # Concatenate
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        
        return embs, pad_masks, att_masks
    
    def embed_suffix(
        self,
        state: Tensor,
        noisy_actions: Tensor,
        timestep: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        编码 suffix (state + noisy actions + time)
        
        Args:
            state: (B, state_dim) 机器人状态
            noisy_actions: (B, n_action_steps, action_dim) 带噪声的动作
            timestep: (B,) 时间步
            
        Returns:
            embs: (B, suffix_len, hidden) 嵌入
            pad_masks: (B, suffix_len) padding mask
            att_masks: 注意力 pattern
        """
        embs = []
        pad_masks = []
        att_masks = []
        device = state.device
        dtype = state.dtype
        bsize = state.shape[0]
        
        # Embed state
        state_emb = self.state_proj(state)
        # 不再手动转换 dtype
        embs.append(state_emb[:, None, :])
        
        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        att_masks += [0]  # State 可以被所有位置 attend
        
        # Embed time
        time_emb = fourier_embed(timestep, self.config.proj_width, device=device)
        
        # Embed actions
        action_emb = self.action_in_proj(noisy_actions)
        
        # Fuse time + action
        time_emb = time_emb[:, None, :].expand(-1, action_emb.shape[1], -1).to(dtype=action_emb.dtype)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)
        
        embs.append(action_time_emb)
        
        action_mask = torch.ones(bsize, action_time_emb.shape[1], dtype=torch.bool, device=device)
        pad_masks.append(action_mask)
        
        # Action tokens: 只有第一个可以 attend prefix
        att_masks += [1] + ([0] * (self.config.n_action_steps - 1))
        
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        
        return embs, pad_masks, att_masks
    
    def forward(
        self,
        images: List[Tensor],
        img_masks: List[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        actions: Tensor,
        noise: Optional[Tensor] = None,
        time: Optional[Tensor] = None,
    ) -> Tensor:
        """
        训练前向传播，计算 Flow Matching loss
        
        Args:
            images: 图像列表
            img_masks: 图像 mask
            lang_tokens: 语言 tokens
            lang_masks: 语言 mask
            state: 机器人状态
            actions: 目标动作
            noise: 可选噪声
            time: 可选时间步
            
        Returns:
            velocity 预测 (用于计算 MSE loss)
        """
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        
        # 计算 noisy actions (linear interpolation)
        time_expanded = time[:, None, None]
        noisy_actions = time_expanded * noise + (1 - time_expanded) * actions
        
        # 目标 velocity: u_t = noise - actions
        u_t = noise - actions
        
        # Embed prefix
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        
        # Embed suffix
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state, noisy_actions, time
        )
        
        # Project suffix to VLM hidden size
        suffix_embs = self.suffix_proj(suffix_embs)
        
        # Concatenate
        embs = torch.cat([prefix_embs, suffix_embs], dim=1)
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = prefix_att_masks + suffix_att_masks
        
        # Build attention mask
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks.long(), dim=1) - 1
        
        # Forward through model
        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        
        # Extract action predictions
        suffix_out = suffix_out[:, -self.config.n_action_steps:]
        # Project from expert hidden to proj_width
        suffix_out = self.expert_out_proj(suffix_out)
        v_t = self.action_out_proj(suffix_out)
        v_t = v_t.to(dtype=torch.float32)
        
        # 计算 MSE loss
        mse_loss = F.mse_loss(v_t, u_t)
        
        return {"MSE": mse_loss, "target": u_t, "prediction": v_t}
    
    def sample_actions(
        self,
        images: List[Tensor],
        img_masks: List[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        """
        采样动作 (推理)
        
        使用 ODE 求解从 noise 到 actions
        
        Args:
            images: 图像列表
            img_masks: 图像 mask
            lang_tokens: 语言 tokens
            lang_masks: 语言 mask
            state: 机器人状态
            noise: 可选初始噪声
            
        Returns:
            (B, n_action_steps, action_dim) 采样的动作
        """
        bsize = state.shape[0]
        device = state.device
        
        if noise is None:
            actions_shape = (bsize, self.config.n_action_steps, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)
        
        # Embed prefix 并缓存 KV
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks.long(), dim=1) - 1
        
        # 计算 KV cache
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        
        # Euler 积分
        dt = -1.0 / self.config.num_steps
        x_t = noise
        
        for step in range(self.config.num_steps):
            t = 1.0 - step / self.config.num_steps
            timestep = torch.full((bsize,), t, device=device)
            
            # 编码 suffix
            suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
                state, x_t, timestep
            )
            
            # Project suffix to VLM hidden size
            suffix_embs = self.suffix_proj(suffix_embs)
            
            # 构建完整 attention mask
            total_len = prefix_pad_masks.shape[1] + suffix_pad_masks.shape[1]
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = prefix_att_masks + suffix_att_masks
            
            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks.long(), dim=1) - 1
            
            # Forward with cache
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=True,
                fill_kv_cache=False,
            )
            
            # 预测 velocity
            suffix_out = suffix_out[:, -self.config.n_action_steps:]
            # Project from expert hidden to proj_width
            suffix_out = self.expert_out_proj(suffix_out)
            v_t = self.action_out_proj(suffix_out)
            v_t = v_t.to(dtype=torch.float32)
            
            # Euler step: x_{t-dt} = x_t + v_t * dt
            x_t = x_t + v_t * dt
        
        return x_t


class PI0Policy(BasePolicy):
    """
    π0 Policy Wrapper
    
    将 PI0FlowMatching 封装为 BasePolicy 接口
    """
    
    def __init__(
        self,
        config: PI0Config,
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        paligemma_config: PaliGemmaWithExpertConfig = None,
        use_tiny: bool = False,
    ):
        state_dim = state_dim or config.max_state_dim
        action_dim = action_dim or config.max_action_dim
        
        super().__init__(state_dim, action_dim)
        
        self.config = config
        
        # 如果使用 tiny 配置
        if use_tiny:
            paligemma_config = PaliGemmaWithExpertConfig.tiny()
        
        self.model = PI0FlowMatching(config, paligemma_config)
        
        # Action queue for chunked actions
        self._action_queue = deque([], maxlen=config.n_action_steps)
        
        # Tokenizer (placeholder - 实际需要加载)
        self.language_tokenizer = None
    
    def reset(self):
        """重置 action queue"""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)
    
    def forward(
        self,
        obs: Dict[str, Tensor],
        robot_state: Tensor,
    ) -> Tensor:
        """
        前向传播 (训练)
        
        Args:
            obs: 包含 images 和 language 的字典
            robot_state: (B, state_dim) 机器人状态
            
        Returns:
            (B, n_action_steps, action_dim) 动作
        """
        images, img_masks = self._prepare_images(obs)
        lang_tokens, lang_masks = self._prepare_language(obs)
        
        # 只返回采样的动作 (推理模式)
        actions = self.model.sample_actions(
            images, img_masks,
            lang_tokens, lang_masks,
            robot_state,
        )
        
        return actions
    
    def compute_loss(
        self,
        obs: Dict[str, Tensor],
        robot_state: Tensor,
        actions: Tensor,
    ) -> Dict[str, Tensor]:
        """
        计算训练 loss
        
        Args:
            obs: 观测字典
            robot_state: 机器人状态
            actions: 目标动作
            
        Returns:
            包含 loss 的字典
        """
        images, img_masks = self._prepare_images(obs)
        lang_tokens, lang_masks = self._prepare_language(obs)
        
        losses = self.model.forward(
            images, img_masks,
            lang_tokens, lang_masks,
            robot_state,
            actions,
        )
        
        return losses
    
    def act(
        self,
        obs: Observation,
        robot_state: RobotState,
        deterministic: bool = True,
    ) -> Action:
        """
        推理接口
        """
        self.eval()
        
        # 如果 queue 为空，重新采样
        if len(self._action_queue) == 0:
            with torch.no_grad():
                # 准备输入
                obs_dict = self._obs_to_dict(obs)
                state_tensor = self._state_to_tensor(robot_state)
                
                # 采样动作
                actions = self.forward(obs_dict, state_tensor)
                
                # 填充 queue
                for i in range(actions.shape[1]):
                    self._action_queue.append(actions[0, i].cpu().numpy())
        
        # 从 queue 取动作
        action_data = self._action_queue.popleft()
        
        return Action(data=action_data, space="joint")
    
    def _prepare_images(self, obs: Dict[str, Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """准备图像输入"""
        images = []
        masks = []
        batch_size = None
        
        for key in self.config.image_features:
            if key in obs:
                img = obs[key]
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                images.append(img)
                batch_size = img.shape[0]
                num_tokens = self.config.resize_imgs_with_padding[0] // 14 * self.config.resize_imgs_with_padding[1] // 14
                masks.append(torch.ones(batch_size, num_tokens, dtype=torch.bool, device=img.device))
        
        # 添加空相机
        if batch_size is None:
            batch_size = 1
        
        for i in range(self.config.empty_cameras):
            device = images[0].device if images else torch.device('cuda')
            empty_img = torch.zeros(batch_size, 3, *self.config.resize_imgs_with_padding, device=device)
            images.append(empty_img)
            masks.append(torch.zeros(batch_size, 256, dtype=torch.bool, device=device))
        
        return images, masks
    
    def _prepare_language(self, obs: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        """准备语言输入"""
        # Placeholder - 实际需要 tokenizer
        device = next(self.parameters()).device
        
        if "language_tokens" in obs:
            return obs["language_tokens"], obs["language_masks"]
        
        # 从 obs 推断 batch_size
        batch_size = 1
        for key in self.config.image_features:
            if key in obs:
                batch_size = obs[key].shape[0]
                break
        
        # 默认返回空 tokens
        tokens = torch.zeros(batch_size, self.config.tokenizer_max_length, dtype=torch.long, device=device)
        masks = torch.ones(batch_size, self.config.tokenizer_max_length, dtype=torch.bool, device=device)
        
        return tokens, masks
    
    def _obs_to_dict(self, obs: Observation) -> Dict[str, Tensor]:
        """将 Observation 转为字典"""
        device = next(self.parameters()).device
        result = {}
        
        for name, img in obs.images.items():
            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor.to(device)
            result[f"observation.{name}"] = img_tensor
        
        result["language"] = obs.language
        
        return result
    
    def _state_to_tensor(self, state: RobotState) -> Tensor:
        """将 RobotState 转为 Tensor"""
        device = next(self.parameters()).device
        state_array = state.to_array()
        
        # Pad to max_state_dim
        if len(state_array) < self.config.max_state_dim:
            padded = np.zeros(self.config.max_state_dim, dtype=np.float32)
            padded[:len(state_array)] = state_array
            state_array = padded
        
        return torch.from_numpy(state_array).float().unsqueeze(0).to(device)
    
    def get_optim_params(self):
        """获取需要优化的参数"""
        return self.parameters()
