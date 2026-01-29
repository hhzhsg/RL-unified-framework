"""
ACT++ 策略

参考 inference2.py 实现，用于 HIL 框架

需要外部 ACT++ 仓库：
  export ACT_PLUS_PLUS_PATH=/path/to/act-plus-plus
  或在 run_act_hil.py 中设置路径
"""
import os
import sys
import pickle
from types import SimpleNamespace
from typing import Dict, Any

import numpy as np
import torch
import torchvision.transforms as transforms

# 从环境变量或默认路径导入 ACT++ detr 模块
ACT_PATH = os.environ.get("ACT_PLUS_PLUS_PATH") or os.environ.get("ACT_PLUS_PLUS", "/home/charles/rl_code/act-plus-plus")
DETR_PATH = os.path.join(ACT_PATH, "detr")

if ACT_PATH not in sys.path:
    sys.path.insert(0, ACT_PATH)
if DETR_PATH not in sys.path:
    sys.path.insert(0, DETR_PATH)

from detr.models import build_ACT_model


class ACTPolicy:
    """
    ACT++ 策略
    
    加载预训练 checkpoint，执行推理
    支持 HIL 框架的权重同步接口（实现 Policy Protocol）
    
    ACT 模型输出 action chunk (num_queries 步动作)，
    本策略支持两种模式：
    1. 每步查询：每个 timestep 都调用模型，取第一个动作
    2. Chunk 查询：每 query_frequency 步查询一次，缓存 chunk 后逐步执行
    """
    
    def __init__(
        self,
        ckpt_dir: str,
        ckpt_name: str = "policy_last.ckpt",
        device: str = "cuda",
        camera_names: list = None,
        action_dim: int = 15,
        state_dim: int = None,  # 从 dataset_stats 自动推断
        query_frequency: int = 1,  # 每 N 步查询一次模型（1 = 每步查询）
        temporal_agg: bool = False,  # 是否使用时序聚合
    ):
        """
        Args:
            ckpt_dir: checkpoint 目录（包含 policy_*.ckpt 和 dataset_stats.pkl）
            ckpt_name: checkpoint 文件名
            device: 设备
            camera_names: 相机名称列表
            action_dim: 动作维度
            state_dim: 状态维度（可选，从 stats 推断）
            query_frequency: 每 N 步查询一次模型（1 = 每步查询，默认）
            temporal_agg: 是否使用时序聚合（多个 chunk 加权平均）
        """
        self._device = torch.device(device)
        self.ckpt_dir = ckpt_dir
        self.action_dim = action_dim
        self.query_frequency = query_frequency
        self.temporal_agg = temporal_agg
        
        # 默认相机配置（与训练时一致）
        self.camera_names = camera_names or [
            'v4l2/cam_high_0/color',
            'v4l2/cam_high_1/color',
            'v4l2/cam_right_wrist_0/color',
            'v4l2/cam_right_wrist_1/color',
        ]
        
        # 加载数据统计（用于归一化）
        stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)
        print(f"[ACTPolicy] Loaded stats from {stats_path}")
        print(f"  qpos_mean shape: {self.stats['qpos_mean'].shape}")
        print(f"  action_mean shape: {self.stats['action_mean'].shape}")
        
        # 推断状态维度
        self.state_dim = state_dim or len(self.stats['qpos_mean'])
        
        # 构建模型
        self.model = self._build_model()
        self.num_queries = self.model.num_queries  # action chunk 长度
        
        # 加载权重
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        self._load_checkpoint(ckpt_path)
        
        # 图像归一化（ImageNet 标准）
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Action chunk 缓存（用于 query_frequency > 1 的情况）
        self._action_chunk = None  # (num_queries, action_dim)
        self._chunk_step = 0  # 当前在 chunk 中的位置
        self._timestep = 0  # 全局时间步
        
        # Temporal aggregation 缓存
        if self.temporal_agg:
            # all_time_actions[t, :] = 在时刻 t 查询到的 action chunk
            self._all_time_actions = None  # 延迟初始化
        
        print(f"[ACTPolicy] Ready! device={device}, action_dim={action_dim}, "
              f"num_queries={self.num_queries}, query_freq={query_frequency}")
    
    def _build_model(self) -> torch.nn.Module:
        """构建 ACT 模型"""
        # 构造 args（与训练时一致）
        args = SimpleNamespace(
            # optimizer / training
            lr=1e-4,
            lr_backbone=1e-5,
            batch_size=2,
            weight_decay=1e-4,
            epochs=300,
            lr_drop=200,
            clip_max_norm=0.1,

            # backbone
            backbone='resnet18',
            dilation=False,
            position_embedding='sine',
            camera_names=self.camera_names,

            # transformer
            enc_layers=4,
            dec_layers=7,
            dim_feedforward=3200,
            hidden_dim=512,
            dropout=0.1,
            nheads=8,
            num_queries=50,
            pre_norm=False,

            # segmentation
            masks=False,

            # imitate / rollout
            eval=False,
            onscreen_render=False,
            kl_weight=None,
            chunk_size=None,
            temporal_agg=False,

            # VQ
            vq=False,
            vq_class=None,
            vq_dim=None,

            # ACT / policy
            load_pretrain=False,
            action_dim=self.action_dim,
            state_dim=self.state_dim,
            no_encoder=False,

            # logging / ckpt
            eval_every=500,
            validate_every=500,
            save_every=500,
            resume_ckpt_path=None,

            # data / misc
            skip_mirrored_data=False,
            actuator_network_dir=None,

            # temporal
            history_len=None,
            future_len=None,
            prediction_len=None,
        )
        
        model = build_ACT_model(args)
        model.to(self._device)
        model.eval()
        
        return model
    
    def _load_checkpoint(self, ckpt_path: str) -> None:
        """加载 checkpoint"""
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        ckpt = torch.load(ckpt_path, map_location=self._device)
        
        # 处理 state_dict 格式差异
        state_dict = ckpt
        
        # 1. 去掉 'model.' 前缀（来自训练时的包装）
        if any(k.startswith('model.') for k in state_dict.keys()):
            print("[ACTPolicy] Removing 'model.' prefix from checkpoint keys...")
            state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
        
        # 2. 处理 encoder 结构差异（旧版 vs 新版 ACT++）
        # 旧版: encoder.layers.0... → 新版: encoder.encoder.layers.0...
        # 检测是否需要映射
        model_keys = set(self.model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        
        # 检测 encoder 结构差异
        needs_encoder_remap = (
            any('encoder.encoder.layers' in k for k in model_keys) and
            any(k.startswith('encoder.layers') for k in ckpt_keys)
        )
        
        if needs_encoder_remap:
            print("[ACTPolicy] Remapping encoder keys (old format → new format)...")
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('encoder.layers'):
                    # encoder.layers.X... → encoder.encoder.layers.X...
                    new_k = k.replace('encoder.layers', 'encoder.encoder.layers', 1)
                elif k.startswith('decoder.'):
                    # decoder.X... → encoder.decoder.X...
                    new_k = 'encoder.' + k
                else:
                    new_k = k
                new_state_dict[new_k] = v
            state_dict = new_state_dict
        
        # 3. 尝试加载（允许不严格匹配）
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"[ACTPolicy] Strict loading failed, trying non-strict...")
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
            if unexpected:
                print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
        
        print(f"[ACTPolicy] Loaded checkpoint from {ckpt_path}")
    
    def _preprocess_qpos(self, qpos: np.ndarray) -> torch.Tensor:
        """qpos 预处理：归一化"""
        qpos_norm = (qpos - self.stats['qpos_mean']) / self.stats['qpos_std']
        return torch.from_numpy(qpos_norm).float().to(self._device).unsqueeze(0)
    
    def _postprocess_action(self, action: torch.Tensor) -> np.ndarray:
        """action 后处理：反归一化"""
        action_np = action.squeeze(0).cpu().numpy()
        return action_np * self.stats['action_std'] + self.stats['action_mean']
    
    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        图像预处理：ImageNet 归一化
        
        输入: 来自 H1RobotEnv，(1, num_cam, C, H, W) 或 (num_cam, C, H, W)，[0,1] on GPU
        输出: (1, num_cam, C, H, W) ImageNet 归一化
        """
        if images.device != self._device:
            images = images.to(self._device)
        
        # 确保是 5D: (B, num_cam, C, H, W)
        if images.dim() == 4:
            images = images.unsqueeze(0)
        
        # ImageNet 归一化（对每个相机图像）
        B, N, C, H, W = images.shape
        images = images.view(B * N, C, H, W)
        images = self.normalize(images)
        images = images.view(B, N, C, H, W)
        
        return images
    
    # ========== PolicyAdapter 接口 ==========
    
    def act(self, obs: Dict[str, Any], deterministic: bool = False) -> np.ndarray:
        """
        推理动作
        
        支持两种模式：
        1. query_frequency=1: 每步都查询模型，取 chunk 第一个动作
        2. query_frequency>1: 每 N 步查询一次，缓存 chunk 后逐步执行
        
        可选 temporal_agg=True 使用时序聚合（多个 chunk 加权平均）
        
        Args:
            obs: 来自 H1RobotEnv.get_observation()
                - qpos: torch.Tensor(1, 15) on GPU
                - images: torch.Tensor(1, 4, C, H, W) on GPU，已归一化到 [0,1]
            deterministic: 是否确定性推理（ACT 默认确定性）
            
        Returns:
            action: np.ndarray(15,) - 右臂(7) + 右gripper(1) + waist(3) + head(2) + base(2)
        """
        qpos = obs.get('qpos')
        images = obs.get('images')
        
        if qpos is None:
            raise ValueError("obs must contain 'qpos'")
        if images is None or images.numel() == 0:
            raise ValueError("obs must contain 'images'")
        
        t = self._timestep
        
        # 是否需要查询模型
        need_query = (t % self.query_frequency == 0) or (self._action_chunk is None)
        
        if need_query:
            # 预处理 qpos
            if isinstance(qpos, torch.Tensor):
                qpos_np = qpos.squeeze(0).cpu().numpy()
            else:
                qpos_np = np.asarray(qpos).flatten()
            qpos_tensor = self._preprocess_qpos(qpos_np)
            
            # 预处理图像
            images_tensor = self._preprocess_images(images)
            
            # 推理
            with torch.no_grad():
                a_hat, _, (_, _), _, _ = self.model(qpos_tensor, images_tensor, env_state=None)
            
            # a_hat: (1, num_queries, action_dim)
            self._action_chunk = a_hat.squeeze(0).cpu().numpy()  # (num_queries, action_dim)
            self._chunk_step = 0
            
            # Temporal aggregation: 存储当前 chunk
            if self.temporal_agg:
                if self._all_time_actions is None:
                    # 延迟初始化
                    max_timesteps = 10000  # 足够大
                    self._all_time_actions = np.zeros(
                        (max_timesteps, max_timesteps + self.num_queries, self.action_dim)
                    )
                # 存储 chunk
                self._all_time_actions[t, t:t+self.num_queries] = self._action_chunk
        
        # 获取当前动作
        if self.temporal_agg and t > 0:
            # 时序聚合：对所有覆盖当前时刻的 chunk 加权平均
            actions_for_curr_step = self._all_time_actions[:t+1, t]
            # 找到非零的（有效的）动作
            actions_populated = np.any(actions_for_curr_step != 0, axis=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]
            
            if len(actions_for_curr_step) > 0:
                # 指数加权（越新的 chunk 权重越大）
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                raw_action = np.sum(actions_for_curr_step * exp_weights[:, None], axis=0)
            else:
                raw_action = self._action_chunk[self._chunk_step]
        else:
            # 直接取 chunk 中对应位置的动作
            raw_action = self._action_chunk[self._chunk_step]
        
        # 反归一化
        action = raw_action * self.stats['action_std'] + self.stats['action_mean']
        
        # 更新计数器
        self._timestep += 1
        self._chunk_step = min(self._chunk_step + 1, self.num_queries - 1)
        
        return action
    
    def get_weights(self) -> Dict[str, torch.Tensor]:
        """获取模型权重（用于 HIL 同步）"""
        return {k: v.cpu() for k, v in self.model.state_dict().items()}
    
    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """加载模型权重（用于 HIL 同步）"""
        self.model.load_state_dict(weights)
        self.model.to(self._device)
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    def reset(self) -> None:
        """重置策略状态：清除 action chunk 缓存"""
        self._action_chunk = None
        self._chunk_step = 0
        self._timestep = 0
        if self.temporal_agg:
            self._all_time_actions = None
