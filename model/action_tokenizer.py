"""
VLA-RL 离散动作 Tokenization

将连续动作离散化为 token，用于:
1. 与 VLM 对接 (动作作为语言 token)
2. π₀.₅ 风格的离散动作训练
3. 加速 VLM 收敛

参考:
- OpenVLA: 256-bin 离散化
- π₀.₅: 离散动作 + 知识隔离
- FAST: 频域压缩

Discretization 方法:
1. Uniform Binning: 均匀划分 [-1, 1] 为 N bins
2. K-Means Binning: 基于数据分布的聚类
3. VQ-VAE: 学习离散 codebook
"""
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ActionTokenizerConfig:
    """动作 Tokenizer 配置"""
    method: str = "uniform"       # "uniform" | "kmeans" | "vqvae"
    num_bins: int = 256           # 每个动作维度的 bin 数量
    action_dim: int = 7           # 动作维度
    action_min: float = -1.0      # 动作最小值
    action_max: float = 1.0       # 动作最大值
    vocab_offset: int = 0         # token 偏移量 (用于与语言 token 共存)


class ActionTokenizer:
    """
    动作 Tokenizer 基类
    
    将连续动作离散化为 token，支持双向转换:
    - encode: continuous action -> discrete tokens
    - decode: discrete tokens -> continuous action
    """
    
    def __init__(self, config: ActionTokenizerConfig):
        self.config = config
        self.num_bins = config.num_bins
        self.action_dim = config.action_dim
        self.action_min = config.action_min
        self.action_max = config.action_max
        self.vocab_offset = config.vocab_offset
        
        # 计算 bin 边界
        self.bin_edges = np.linspace(
            self.action_min, self.action_max, self.num_bins + 1
        )
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
    
    def encode(self, actions: np.ndarray) -> np.ndarray:
        """
        将连续动作编码为离散 token
        
        Args:
            actions: (B, action_dim) 或 (action_dim,) 连续动作
            
        Returns:
            tokens: (B, action_dim) 或 (action_dim,) 离散 token
        """
        # 确保在范围内
        actions = np.clip(actions, self.action_min, self.action_max)
        
        # 计算 bin index
        # 使用 digitize 找到每个值所属的 bin
        tokens = np.digitize(actions, self.bin_edges[1:-1])
        
        # 添加 vocab offset
        tokens = tokens + self.vocab_offset
        
        return tokens.astype(np.int64)
    
    def decode(self, tokens: np.ndarray) -> np.ndarray:
        """
        将离散 token 解码为连续动作
        
        Args:
            tokens: (B, action_dim) 或 (action_dim,) 离散 token
            
        Returns:
            actions: (B, action_dim) 或 (action_dim,) 连续动作
        """
        # 移除 vocab offset
        tokens = tokens - self.vocab_offset
        
        # 确保 token 在有效范围内
        tokens = np.clip(tokens, 0, self.num_bins - 1)
        
        # 取 bin center 作为动作值
        actions = self.bin_centers[tokens]
        
        return actions.astype(np.float32)
    
    def encode_tensor(self, actions: torch.Tensor) -> torch.Tensor:
        """Tensor 版本的 encode"""
        actions_np = actions.cpu().numpy()
        tokens_np = self.encode(actions_np)
        return torch.from_numpy(tokens_np).to(actions.device)
    
    def decode_tensor(self, tokens: torch.Tensor) -> torch.Tensor:
        """Tensor 版本的 decode"""
        tokens_np = tokens.cpu().numpy()
        actions_np = self.decode(tokens_np)
        return torch.from_numpy(actions_np).to(tokens.device)
    
    @property
    def vocab_size(self) -> int:
        """词表大小"""
        return self.num_bins * self.action_dim + self.vocab_offset


class LearnedActionTokenizer(nn.Module):
    """
    可学习的动作 Tokenizer (VQ-VAE 风格)
    
    学习一个离散 codebook，将连续动作映射到最近的 code
    
    优势:
    - 自适应数据分布
    - 可端到端训练
    - 支持 action chunk
    """
    
    def __init__(self, 
                 action_dim: int,
                 codebook_size: int = 256,
                 embedding_dim: int = 64,
                 commitment_cost: float = 0.25):
        """
        Args:
            action_dim: 动作维度
            codebook_size: codebook 大小
            embedding_dim: 嵌入维度
            commitment_cost: commitment loss 系数
        """
        super().__init__()
        
        self.action_dim = action_dim
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Encoder: action -> embedding
        self.encoder = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )
        
        # Codebook
        self.codebook = nn.Embedding(codebook_size, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        
        # Decoder: embedding -> action
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),  # 输出 [-1, 1]
        )
    
    def encode(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码动作为 token
        
        Args:
            actions: (B, action_dim) 连续动作
            
        Returns:
            tokens: (B,) 离散 token
            embeddings: (B, embedding_dim) 量化后的嵌入
        """
        # Encode
        z = self.encoder(actions)
        
        # 找最近的 codebook entry
        distances = torch.cdist(z, self.codebook.weight)
        tokens = distances.argmin(dim=-1)
        
        # 获取量化嵌入
        embeddings = self.codebook(tokens)
        
        return tokens, embeddings
    
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        解码 token 为动作
        
        Args:
            tokens: (B,) 离散 token
            
        Returns:
            actions: (B, action_dim) 连续动作
        """
        embeddings = self.codebook(tokens)
        actions = self.decoder(embeddings)
        return actions
    
    def forward(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播 (训练时)
        
        Args:
            actions: (B, action_dim) 连续动作
            
        Returns:
            reconstructed: (B, action_dim) 重建的动作
            tokens: (B,) 离散 token
            losses: 包含各项 loss 的字典
        """
        # Encode
        z = self.encoder(actions)
        
        # 量化
        distances = torch.cdist(z, self.codebook.weight)
        tokens = distances.argmin(dim=-1)
        z_q = self.codebook(tokens)
        
        # Straight-through estimator
        z_q_st = z + (z_q - z).detach()
        
        # Decode
        reconstructed = self.decoder(z_q_st)
        
        # 计算 losses
        reconstruction_loss = F.mse_loss(reconstructed, actions)
        commitment_loss = F.mse_loss(z, z_q.detach())
        codebook_loss = F.mse_loss(z.detach(), z_q)
        
        losses = {
            "reconstruction_loss": reconstruction_loss,
            "commitment_loss": self.commitment_cost * commitment_loss,
            "codebook_loss": codebook_loss,
            "total_loss": reconstruction_loss + self.commitment_cost * commitment_loss + codebook_loss,
        }
        
        return reconstructed, tokens, losses


class ActionChunkTokenizer:
    """
    Action Chunk Tokenizer
    
    将多步动作 (action chunk) 一起离散化
    用于 π₀.₅ 风格的 action chunk 预测
    """
    
    def __init__(self, 
                 action_dim: int,
                 chunk_size: int = 4,
                 num_bins: int = 256):
        """
        Args:
            action_dim: 单步动作维度
            chunk_size: chunk 大小 (步数)
            num_bins: 每个动作维度的 bin 数量
        """
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.num_bins = num_bins
        
        # 每步的 tokenizer
        self.step_tokenizer = ActionTokenizer(
            ActionTokenizerConfig(
                num_bins=num_bins,
                action_dim=action_dim,
            )
        )
    
    def encode(self, action_chunk: np.ndarray) -> np.ndarray:
        """
        编码 action chunk
        
        Args:
            action_chunk: (B, chunk_size, action_dim) 或 (chunk_size, action_dim)
            
        Returns:
            tokens: (B, chunk_size * action_dim) 或 (chunk_size * action_dim,)
        """
        if action_chunk.ndim == 2:
            # (chunk_size, action_dim) -> (chunk_size * action_dim,)
            flat = action_chunk.reshape(-1)
            tokens = self.step_tokenizer.encode(flat)
            return tokens
        else:
            # (B, chunk_size, action_dim) -> (B, chunk_size * action_dim)
            B = action_chunk.shape[0]
            flat = action_chunk.reshape(B, -1)
            tokens = self.step_tokenizer.encode(flat)
            return tokens
    
    def decode(self, tokens: np.ndarray) -> np.ndarray:
        """
        解码 action chunk
        
        Args:
            tokens: (B, chunk_size * action_dim) 或 (chunk_size * action_dim,)
            
        Returns:
            action_chunk: (B, chunk_size, action_dim) 或 (chunk_size, action_dim)
        """
        if tokens.ndim == 1:
            actions = self.step_tokenizer.decode(tokens)
            return actions.reshape(self.chunk_size, self.action_dim)
        else:
            B = tokens.shape[0]
            actions = self.step_tokenizer.decode(tokens)
            return actions.reshape(B, self.chunk_size, self.action_dim)
    
    @property
    def num_tokens_per_chunk(self) -> int:
        """每个 chunk 的 token 数量"""
        return self.chunk_size * self.action_dim


# ==================== 便捷函数 ====================

def create_action_tokenizer(
    method: str = "uniform",
    action_dim: int = 7,
    num_bins: int = 256,
    **kwargs
) -> ActionTokenizer:
    """
    创建动作 Tokenizer
    
    Args:
        method: 离散化方法 ("uniform" | "learned")
        action_dim: 动作维度
        num_bins: bin 数量
        
    Returns:
        ActionTokenizer 实例
    """
    if method == "uniform":
        config = ActionTokenizerConfig(
            method="uniform",
            num_bins=num_bins,
            action_dim=action_dim,
            **kwargs
        )
        return ActionTokenizer(config)
    
    elif method == "learned":
        return LearnedActionTokenizer(
            action_dim=action_dim,
            codebook_size=num_bins,
            **kwargs
        )
    
    else:
        raise ValueError(f"未知方法: {method}")


# ==================== 测试 ====================

def test_action_tokenizer():
    """测试动作 Tokenizer"""
    print("Testing ActionTokenizer...")
    
    # 创建 tokenizer
    tokenizer = create_action_tokenizer(
        method="uniform",
        action_dim=7,
        num_bins=256,
    )
    
    # 测试 encode/decode
    actions = np.random.uniform(-1, 1, (32, 7)).astype(np.float32)
    
    tokens = tokenizer.encode(actions)
    reconstructed = tokenizer.decode(tokens)
    
    # 计算重建误差
    error = np.abs(actions - reconstructed).mean()
    
    print(f"  Actions shape: {actions.shape}")
    print(f"  Tokens shape: {tokens.shape}")
    print(f"  Token range: [{tokens.min()}, {tokens.max()}]")
    print(f"  Reconstruction error: {error:.6f}")
    print(f"  Expected error (uniform): ~{1/256/2:.6f}")
    
    assert error < 0.01, "重建误差过大"
    print("  ✓ Test passed!")


if __name__ == "__main__":
    test_action_tokenizer()
