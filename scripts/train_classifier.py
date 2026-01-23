#!/usr/bin/env python
"""
Reward Classifier 训练脚本

从采集的成功/失败数据中训练二分类器。
支持两种数据格式：
1. HIL-SERL 格式：record_classifier_data.py 生成的 pickle 文件
2. HDF5 格式：从 demo 数据中提取成功帧

使用示例:
    # 使用 pickle 数据训练（HIL-SERL 格式）
    python scripts/train_classifier.py \
        --success_data classifier_data/h1_200_success.pkl \
        --failure_data classifier_data/h1_failure.pkl \
        --output checkpoints/classifier.pt
    
    # 使用 HDF5 demo 数据训练
    python scripts/train_classifier.py \
        --demo data/demo/h1_demo.hdf5 \
        --output checkpoints/classifier.pt \
        --success_frames 5
"""
import argparse
import sys
import h5py
import pickle as pkl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import Logger


class RewardClassifier(nn.Module):
    """
    基于图像的奖励分类器
    
    输入：多个相机图像（拼接后）
    输出：成功概率 [0, 1]
    
    架构：简单 CNN + MLP（可替换为更复杂的 ViT 等）
    """
    
    def __init__(
        self,
        image_channels: int = 3,
        image_size: Tuple[int, int] = (84, 84),
        num_cameras: int = 2,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.num_cameras = num_cameras
        self.image_size = image_size
        
        # 每个相机的 encoder（共享权重）
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # 计算 encoder 输出维度
        with torch.no_grad():
            dummy = torch.zeros(1, image_channels, *image_size)
            encoder_out_dim = self.encoder(dummy).shape[1]
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(encoder_out_dim * num_cameras, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, images: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            images: List of (B, C, H, W) tensors, one per camera
            
        Returns:
            logits: (B, 1) success logits
        """
        features = []
        for img in images:
            feat = self.encoder(img)
            features.append(feat)
        
        combined = torch.cat(features, dim=-1)
        logits = self.classifier(combined)
        return logits
    
    def predict_reward(
        self,
        images: List[torch.Tensor],
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        预测奖励（二值化）
        
        Args:
            images: List of image tensors
            threshold: 成功阈值
            
        Returns:
            rewards: (B,) 0 或 1
        """
        with torch.no_grad():
            logits = self.forward(images)
            probs = torch.sigmoid(logits).squeeze(-1)
            rewards = (probs > threshold).float()
        return rewards
    
    def predict_proba(self, images: List[torch.Tensor]) -> torch.Tensor:
        """预测成功概率"""
        with torch.no_grad():
            logits = self.forward(images)
            probs = torch.sigmoid(logits).squeeze(-1)
        return probs


class PickleClassifierDataset(Dataset):
    """
    从 pickle 文件加载的 Classifier 数据集（HIL-SERL 格式）
    
    数据格式（由 record_classifier_data.py 生成）：
    - success_data.pkl: List of transitions，每个包含 observations (images dict)
    - failure_data.pkl: List of transitions，每个包含 observations (images dict)
    """
    
    def __init__(
        self,
        success_path: str,
        failure_path: str,
        camera_keys: List[str],
        image_size: Tuple[int, int] = (84, 84),
        balance: bool = True,
    ):
        self.camera_keys = camera_keys
        self.image_size = image_size
        self.samples = []  # List of (images_dict, label)
        
        # 加载成功样本
        with open(success_path, "rb") as f:
            successes = pkl.load(f)
        
        for t in successes:
            # observations 是 images dict 或包含 images 的 dict
            obs = t.get("observations", t.get("next_observations", {}))
            images = self._extract_images(obs)
            if images:
                self.samples.append((images, 1))
        
        num_successes = len(self.samples)
        
        # 加载失败样本
        with open(failure_path, "rb") as f:
            failures = pkl.load(f)
        
        # 平衡采样（可选）
        if balance and len(failures) > num_successes:
            indices = np.random.choice(len(failures), size=num_successes, replace=False)
            failures = [failures[i] for i in indices]
        
        for t in failures:
            obs = t.get("observations", t.get("next_observations", {}))
            images = self._extract_images(obs)
            if images:
                self.samples.append((images, 0))
        
        print(f"[Dataset] Loaded {len(self.samples)} samples from pickle")
        print(f"[Dataset] Positive: {num_successes}")
        print(f"[Dataset] Negative: {len(self.samples) - num_successes}")
    
    def _extract_images(self, obs: Dict) -> Optional[Dict[str, np.ndarray]]:
        """从 observations 中提取图像"""
        images = {}
        for cam in self.camera_keys:
            if cam in obs:
                images[cam] = np.array(obs[cam])
            elif "images" in obs and cam in obs["images"]:
                images[cam] = np.array(obs["images"][cam])
        
        return images if len(images) == len(self.camera_keys) else None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
        images_dict, label = self.samples[idx]
        
        images = []
        for cam in self.camera_keys:
            img = images_dict[cam]
            img = self._preprocess_image(img)
            images.append(img)
        
        label = torch.tensor(label, dtype=torch.float32)
        return images, label
    
    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """预处理图像"""
        # HWC -> CHW
        if img.ndim == 3 and img.shape[-1] in [1, 3, 4]:
            img = np.transpose(img, (2, 0, 1))
        
        img = torch.from_numpy(img).float()
        if img.shape[-2:] != self.image_size:
            img = F.interpolate(
                img.unsqueeze(0),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        
        # Normalize to [0, 1]
        if img.max() > 1.0:
            img = img / 255.0
        
        return img


class HDF5ClassifierDataset(Dataset):
    """
    从 HDF5 demo 文件加载的 Classifier 数据集
    
    从 demo HDF5 中提取：
    - 正样本：每个 episode 的最后 N 帧（成功状态）
    - 负样本：每个 episode 的前面若干帧（非成功状态）
    """
    
    def __init__(
        self,
        hdf5_path: str,
        camera_keys: List[str],
        success_frames: int = 5,
        negative_ratio: float = 1.0,
        image_size: Tuple[int, int] = (84, 84),
    ):
        self.camera_keys = camera_keys
        self.image_size = image_size
        
        self.samples = []  # List of (images_dict, label)
        
        with h5py.File(hdf5_path, "r") as f:
            num_episodes = f.attrs.get("num_episodes", 0)
            
            for ep_idx in range(num_episodes):
                ep_key = f"episode_{ep_idx}"
                if ep_key not in f:
                    continue
                
                ep = f[ep_key]
                
                # 检查是否有图像数据
                images_available = all(
                    f"images/{cam}" in ep for cam in camera_keys
                )
                
                if not images_available:
                    print(f"[Warning] Episode {ep_idx} missing images, using obs instead")
                    continue
                
                # 获取 episode 长度
                ep_len = len(ep["action"])
                
                # 正样本：最后 N 帧
                for i in range(max(0, ep_len - success_frames), ep_len):
                    images = {
                        cam: ep[f"images/{cam}"][i] for cam in camera_keys
                    }
                    self.samples.append((images, 1))
                
                # 负样本：前面的帧（按比例采样）
                num_negatives = int(success_frames * negative_ratio)
                negative_indices = np.random.choice(
                    max(0, ep_len - success_frames),
                    size=min(num_negatives, max(0, ep_len - success_frames)),
                    replace=False,
                ) if ep_len > success_frames else []
                
                for i in negative_indices:
                    images = {
                        cam: ep[f"images/{cam}"][i] for cam in camera_keys
                    }
                    self.samples.append((images, 0))
        
        print(f"[Dataset] Loaded {len(self.samples)} samples")
        print(f"[Dataset] Positive: {sum(1 for _, l in self.samples if l == 1)}")
        print(f"[Dataset] Negative: {sum(1 for _, l in self.samples if l == 0)}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
        images_dict, label = self.samples[idx]
        
        # 预处理图像
        images = []
        for cam in self.camera_keys:
            img = images_dict[cam]
            img = self._preprocess_image(img)
            images.append(img)
        
        label = torch.tensor(label, dtype=torch.float32)
        return images, label
    
    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """预处理图像"""
        # HWC -> CHW
        if img.ndim == 3 and img.shape[-1] in [1, 3, 4]:
            img = np.transpose(img, (2, 0, 1))
        
        # Resize
        img = torch.from_numpy(img).float()
        if img.shape[-2:] != self.image_size:
            img = F.interpolate(
                img.unsqueeze(0),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        
        # Normalize to [0, 1]
        if img.max() > 1.0:
            img = img / 255.0
        
        return img


def collate_fn(batch):
    """自定义 collate 函数处理多相机图像"""
    images_list = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    
    # 按相机索引重组
    num_cameras = len(images_list[0])
    batched_images = []
    for cam_idx in range(num_cameras):
        cam_batch = torch.stack([imgs[cam_idx] for imgs in images_list])
        batched_images.append(cam_batch)
    
    return batched_images, labels


def train_classifier(
    model: RewardClassifier,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int,
    lr: float,
    device: torch.device,
    logger: Logger,
) -> Dict[str, List[float]]:
    """训练分类器"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images = [img.to(device) for img in images]
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(images).squeeze(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = [img.to(device) for img in images]
                    labels = labels.to(device)
                    
                    logits = model(images).squeeze(-1)
                    loss = criterion(logits, labels)
                    
                    val_loss += loss.item() * labels.size(0)
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            
            val_loss /= val_total
            val_acc = val_correct / val_total
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            
            logger.info(
                f"Epoch {epoch + 1}/{epochs}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.2%}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.2%}"
            )
        else:
            logger.info(
                f"Epoch {epoch + 1}/{epochs}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.2%}"
            )
    
    return history


def main():
    parser = argparse.ArgumentParser(description="Train Reward Classifier")
    
    # 数据源（二选一）
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--demo", type=str, help="Demo HDF5 path (auto-extract success frames)")
    data_group.add_argument("--success_data", type=str, 
                            help="Success pickle path (HIL-SERL format)")
    
    parser.add_argument("--failure_data", type=str, 
                        help="Failure pickle path (required with --success_data)")
    parser.add_argument("--output", type=str, default="checkpoints/classifier.pt",
                        help="Output model path")
    parser.add_argument("--cameras", type=str, nargs="+", 
                        default=["cam_high", "cam_left_wrist", "cam_right_wrist"],
                        help="Camera names to use (H1: cam_high, cam_left_wrist, cam_right_wrist)")
    parser.add_argument("--success_frames", type=int, default=5,
                        help="Number of final frames to use as positive samples (HDF5 mode)")
    parser.add_argument("--negative_ratio", type=float, default=1.0,
                        help="Ratio of negative to positive samples")
    parser.add_argument("--balance", action="store_true", default=True,
                        help="Balance positive and negative samples")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--image_size", type=int, default=84, help="Image size")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split")
    args = parser.parse_args()
    
    # 校验参数
    if args.success_data and not args.failure_data:
        parser.error("--failure_data is required when using --success_data")
    
    logger = Logger(log_dir="./logs")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 创建数据集
    if args.success_data:
        # HIL-SERL pickle 格式
        logger.info(f"Loading from pickle: {args.success_data}, {args.failure_data}")
        logger.info(f"Cameras: {args.cameras}")
        
        dataset = PickleClassifierDataset(
            success_path=args.success_data,
            failure_path=args.failure_data,
            camera_keys=args.cameras,
            image_size=(args.image_size, args.image_size),
            balance=args.balance,
        )
    else:
        # HDF5 demo 格式
        logger.info(f"Loading from HDF5: {args.demo}")
        logger.info(f"Cameras: {args.cameras}")
        logger.info(f"Success frames: {args.success_frames}")
        
        dataset = HDF5ClassifierDataset(
            hdf5_path=args.demo,
            camera_keys=args.cameras,
            success_frames=args.success_frames,
            negative_ratio=args.negative_ratio,
            image_size=(args.image_size, args.image_size),
        )
    
    if len(dataset) == 0:
        logger.error("No samples found in dataset. Check if data has image data.")
        return
    
    # 划分训练/验证集
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    ) if val_size > 0 else None
    
    # 创建模型
    model = RewardClassifier(
        image_channels=3,
        image_size=(args.image_size, args.image_size),
        num_cameras=len(args.cameras),
    ).to(device)
    
    logger.info(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    
    # 训练
    history = train_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        logger=logger,
    )
    
    # 保存模型
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "cameras": args.cameras,
            "image_size": args.image_size,
            "num_cameras": len(args.cameras),
        },
        "history": history,
    }, args.output)
    
    logger.info(f"Model saved to {args.output}")
    
    # 打印最终结果
    if history["val_acc"]:
        logger.info(f"Final validation accuracy: {history['val_acc'][-1]:.2%}")
    else:
        logger.info(f"Final training accuracy: {history['train_acc'][-1]:.2%}")


if __name__ == "__main__":
    main()
