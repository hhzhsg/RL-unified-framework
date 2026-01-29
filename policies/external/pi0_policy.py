"""
Pi0 / VLA 模型策略示例

展示如何将大型 VLA 模型（如 pi0.5、OpenVLA）接入 HIL 框架

关键点：
1. 只同步 LoRA/Adapter 参数（减少通信开销）
2. 处理多模态输入（图像 + 语言）
3. 实现 Policy Protocol 的推理接口
"""
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import numpy as np


# ============ 权重同步模式 ============

class WeightSyncMode:
    """权重同步模式"""
    FULL = "full"           # 全量同步
    LORA = "lora"           # 只同步 LoRA 参数
    ADAPTER = "adapter"     # 只同步 Adapter 参数


def filter_weights_by_keyword(
    state_dict: Dict[str, torch.Tensor],
    keywords: list,
    include: bool = True
) -> Dict[str, torch.Tensor]:
    """按关键词过滤权重"""
    result = {}
    for key, value in state_dict.items():
        matches = any(kw in key.lower() for kw in keywords)
        if (include and matches) or (not include and not matches):
            result[key] = value
    return result


class Pi0Policy:
    """
    Pi0 策略（Actor 端）
    
    实现 Policy Protocol，支持：
    - LoRA 参数同步
    - VLA 多模态输入
    - pi0 推理接口
    
    使用示例：
        from some_pi0_library import Pi0Model
        
        model = Pi0Model.from_pretrained("pi0-base")
        policy = Pi0Policy(
            model=model,
            sync_mode="lora",
            language_instruction="pick up the red block",
        )
        
        actor = HILActorLoop(policy, env, config)
    """
    
    def __init__(
        self,
        model: nn.Module,
        sync_mode: str = WeightSyncMode.LORA,
        language_instruction: str = "",
        camera_keys: List[str] = None,
        action_key: str = "action",
        device: str = "cuda",
    ):
        """
        Args:
            model: Pi0 模型实例
            sync_mode: 权重同步模式（"lora" | "adapter" | "full"）
            language_instruction: 语言指令
            camera_keys: 相机图像键名列表
            action_key: 动作输出键名
            device: 设备
        """
        self.model = model
        self.sync_mode = sync_mode
        self.language_instruction = language_instruction
        self.camera_keys = camera_keys or ["images.front", "images.wrist"]
        self.action_key = action_key
        self._device = torch.device(device)
        
        # 同步关键词（根据 sync_mode）
        self._sync_keywords = self._get_sync_keywords()
        
        # 移动模型到设备
        self.model.to(self._device)
        self.model.eval()
    
    def _get_sync_keywords(self) -> List[str]:
        """获取需要同步的参数关键词"""
        if self.sync_mode == WeightSyncMode.LORA:
            return ["lora", "lora_a", "lora_b"]
        elif self.sync_mode == WeightSyncMode.ADAPTER:
            return ["adapter", "adaptor"]
        elif self.sync_mode == WeightSyncMode.HEAD:
            return ["head", "output", "action_head"]
        else:
            return []  # 全量同步
    
    def act(self, obs: Dict[str, Any], deterministic: bool = False) -> np.ndarray:
        """
        推理动作
        
        处理 VLA 多模态输入
        """
        with torch.no_grad():
            # 1. 预处理观测
            model_input = self._preprocess_obs(obs)
            
            # 2. 模型推理
            # 注意：这里需要根据实际 pi0 API 调整
            output = self.model(
                images=model_input["images"],
                language=model_input["language"],
                proprio=model_input.get("proprio"),
            )
            
            # 3. 提取动作
            if isinstance(output, dict):
                action = output[self.action_key]
            else:
                action = output
            
            # 4. 后处理
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            
            if action.ndim > 1:
                action = action.squeeze(0)
            
            return action
    
    def _preprocess_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """预处理观测为模型输入格式"""
        result = {}
        
        # 处理图像
        images = []
        for key in self.camera_keys:
            if key in obs:
                img = obs[key]
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img).float()
                if img.dim() == 3:
                    img = img.unsqueeze(0)  # 添加 batch 维度
                img = img.to(self._device)
                images.append(img)
        
        if images:
            result["images"] = torch.cat(images, dim=1) if len(images) > 1 else images[0]
        
        # 处理语言指令
        result["language"] = self.language_instruction
        
        # 处理本体感知
        if "state" in obs or "proprio" in obs:
            proprio = obs.get("state", obs.get("proprio"))
            if isinstance(proprio, np.ndarray):
                proprio = torch.from_numpy(proprio).float()
            result["proprio"] = proprio.to(self._device).unsqueeze(0)
        
        return result
    
    def get_weights(self) -> Dict[str, torch.Tensor]:
        """获取需要同步的权重（只返回 LoRA/Adapter）"""
        state_dict = self.model.state_dict()
        
        if self.sync_mode == WeightSyncMode.FULL:
            return {k: v.cpu() for k, v in state_dict.items()}
        else:
            filtered = filter_weights_by_keyword(state_dict, self._sync_keywords, include=True)
            return {k: v.cpu() for k, v in filtered.items()}
    
    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """加载权重"""
        # 移动到设备
        weights = {k: v.to(self._device) for k, v in weights.items()}
        
        if self.sync_mode == WeightSyncMode.FULL:
            self.model.load_state_dict(weights)
        else:
            # 部分加载
            current_state = self.model.state_dict()
            current_state.update(weights)
            self.model.load_state_dict(current_state)
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    def set_language_instruction(self, instruction: str) -> None:
        """动态更新语言指令"""
        self.language_instruction = instruction


class Pi0Trainer:
    """
    Pi0 训练器（Learner 端）
    
    特点：
    - 支持 LoRA 微调
    - 处理 VLA 多模态训练
    - 只更新/同步 LoRA 参数
    
    使用示例：
        trainer = Pi0Trainer(
            model=pi0_model,
            sync_mode="lora",
            learning_rate=1e-4,
        )
        
        learner = HILLearnerLoop(trainer, config)
    """
    
    def __init__(
        self,
        model: nn.Module,
        sync_mode: str = WeightSyncMode.LORA,
        learning_rate: float = 1e-4,
        device: str = "cuda",
    ):
        self.model = model
        self.sync_mode = sync_mode
        self._device = torch.device(device)
        
        self._sync_keywords = self._get_sync_keywords()
        
        # 移动模型到设备
        self.model.to(self._device)
        
        # 冻结非 LoRA 参数
        if sync_mode != WeightSyncMode.FULL:
            self._freeze_non_lora_params()
        
        # 优化器（只优化可训练参数）
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
        
        self._train_step = 0
    
    def _get_sync_keywords(self) -> List[str]:
        if self.sync_mode == WeightSyncMode.LORA:
            return ["lora", "lora_a", "lora_b"]
        elif self.sync_mode == WeightSyncMode.ADAPTER:
            return ["adapter", "adaptor"]
        else:
            return []
    
    def _freeze_non_lora_params(self) -> None:
        """冻结非 LoRA 参数"""
        for name, param in self.model.named_parameters():
            if not any(kw in name.lower() for kw in self._sync_keywords):
                param.requires_grad = False
    
    def act(self, obs: Dict[str, Any], deterministic: bool = False) -> np.ndarray:
        """推理（使用 Pi0Policy）"""
        raise NotImplementedError("Use Pi0Policy for inference")
    
    def get_weights(self) -> Dict[str, torch.Tensor]:
        """获取需要同步的权重"""
        state_dict = self.model.state_dict()
        
        if self.sync_mode == WeightSyncMode.FULL:
            return {k: v.cpu() for k, v in state_dict.items()}
        else:
            filtered = filter_weights_by_keyword(state_dict, self._sync_keywords, include=True)
            return {k: v.cpu() for k, v in filtered.items()}
    
    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """加载权重"""
        weights = {k: v.to(self._device) for k, v in weights.items()}
        current_state = self.model.state_dict()
        current_state.update(weights)
        self.model.load_state_dict(current_state)
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播"""
        return self.model(obs)
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]):
        """计算损失"""
        # 这里需要根据实际 pi0 训练逻辑实现
        raise NotImplementedError("Implement based on pi0 training logic")
    
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        执行一步更新
        
        这里需要根据实际 pi0 训练逻辑实现
        """
        self.model.train()
        
        # 示例：简单的 BC 损失
        # 实际应根据 pi0 的训练目标实现
        obs = batch["obs"]
        action = batch["action"]
        
        # 前向传播
        pred_action = self.model(obs)
        
        # 计算损失
        loss = torch.nn.functional.mse_loss(pred_action, action)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self._train_step += 1
        
        return {"loss": loss.item()}
    
    def get_optimizer(self) -> torch.optim.Optimizer:
        return self.optimizer
    
    def save(self, path: str) -> None:
        """保存（只保存 LoRA 参数）"""
        weights = self.get_weights()
        torch.save({
            "weights": weights,
            "train_step": self._train_step,
            "sync_mode": self.sync_mode,
        }, path)
    
    def load(self, path: str) -> None:
        """加载"""
        ckpt = torch.load(path, map_location=self._device)
        self.load_weights(ckpt["weights"])
        self._train_step = ckpt.get("train_step", 0)


# ============ 工厂函数 ============

def create_pi0_policy_and_trainer(
    model: nn.Module,
    sync_mode: str = "lora",
    language_instruction: str = "",
    learning_rate: float = 1e-4,
    device: str = "cuda",
):
    """
    创建 Pi0 的 Policy 和 Trainer
    
    Returns:
        (policy, trainer)
    """
    policy = Pi0Policy(
        model=model,
        sync_mode=sync_mode,
        language_instruction=language_instruction,
        device=device,
    )
    
    trainer = Pi0Trainer(
        model=model,
        sync_mode=sync_mode,
        learning_rate=learning_rate,
        device=device,
    )
    
    return policy, trainer


# 兼容旧名称
Pi0PolicyAdapter = Pi0Policy
Pi0TrainerAdapter = Pi0Trainer
create_pi0_adapters = create_pi0_policy_and_trainer
