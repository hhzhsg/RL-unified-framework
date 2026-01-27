#!/usr/bin/env python
"""
Reward Classifier 训练脚本

参考 HIL-SERL 实现：
- 使用预训练 ResNet-18（ImageNet）作为图像编码器
- 冻结 backbone，只训练 pooling + 分类头
- 支持多相机图像拼接
- Random Crop 数据增强

注意：Classifier 需要人工标注的 success/failure 数据，不支持从 HDF5 demo 自动提取。
请使用 record_classifier_data.py 采集标注数据。

使用示例:
    # 方式 1: 指定数据目录（自动匹配 success/failure 文件）
    python scripts/train_classifier.py \
        --data_dir classifier_data/ \
        --output checkpoints/classifier.pt
    
    # 方式 2: 手动指定文件
    python scripts/train_classifier.py \
        --success_data classifier_data/h1_200_success.pkl \
        --failure_data classifier_data/h1_failure.pkl \
        --output checkpoints/classifier.pt
        
    # 指定相机和预训练模型
    python scripts/train_classifier.py \
        --data_dir classifier_data/ \
        --cameras cam_high cam_left_wrist cam_right_wrist \
        --encoder resnet18 \
        --freeze_encoder
"""
import argparse
import sys
import glob
import pickle as pkl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import Logger


# ============ 数据加载辅助函数 ============

def find_classifier_data(data_dir: str) -> Tuple[List[str], List[str]]:
    """
    从目录中自动匹配 success/failure 数据文件
    
    Args:
        data_dir: 数据目录
        
    Returns:
        (success_files, failure_files): 匹配到的文件列表
    """
    data_dir = Path(data_dir)
    
    # 查找所有 pkl 文件
    all_files = list(data_dir.glob("*.pkl"))
    
    success_files = [str(f) for f in all_files if "success" in f.name.lower()]
    failure_files = [str(f) for f in all_files if "failure" in f.name.lower() or "fail" in f.name.lower()]
    
    return sorted(success_files), sorted(failure_files)


def load_pickle_data(file_paths: List[str]) -> List[Dict]:
    """加载多个 pickle 文件并合并"""
    all_data = []
    for path in file_paths:
        with open(path, "rb") as f:
            data = pkl.load(f)
            all_data.extend(data)
        print(f"[Data] Loaded {len(data)} samples from {path}")
    return all_data


# ============ 预训练 ResNet Encoder ============

class PretrainedResNetEncoder(nn.Module):
    """
    预训练 ResNet 编码器（参考 HIL-SERL）
    
    使用 ImageNet 预训练权重，支持冻结 backbone
    """
    
    def __init__(
        self,
        model_name: str = "resnet18",
        freeze: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()
        
        # 加载预训练模型
        if model_name == "resnet10":
            # ResNet-10 需要自定义（HIL-SERL 使用）
            # 这里用 ResNet-18 的前几层近似
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            self.encoder = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
            )
            self.out_channels = 128
            self.out_size = 28  # 对于 224x224 输入
        elif model_name == "resnet18":
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # 去掉 avgpool 和 fc
            self.out_channels = 512
            self.out_size = 7  # 对于 224x224 输入
        elif model_name == "resnet34":
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
            self.encoder = nn.Sequential(*list(resnet.children())[:-2])
            self.out_channels = 512
            self.out_size = 7
        elif model_name == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            self.encoder = nn.Sequential(*list(resnet.children())[:-2])
            self.out_channels = 2048
            self.out_size = 7
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # 冻结 backbone（HIL-SERL 默认冻结）
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 图像
        Returns:
            features: (B, C', H', W') 特征图
        """
        return self.encoder(x)


class SpatialLearnedEmbeddings(nn.Module):
    """
    Spatial Learned Embeddings（参考 HIL-SERL）
    
    学习空间位置的特征表示，比 avg pooling 保留更多空间信息
    """
    
    def __init__(self, height: int, width: int, channels: int, num_features: int = 8):
        super().__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.num_features = num_features
        
        # 学习的空间嵌入
        self.embeddings = nn.Parameter(
            torch.randn(num_features, height, width) * 0.01
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C * num_features)
        """
        B, C, H, W = x.shape
        
        # Softmax over spatial dimensions
        weights = self.embeddings.view(self.num_features, -1)
        weights = F.softmax(weights, dim=-1)
        weights = weights.view(self.num_features, self.height, self.width)
        
        # 加权求和
        features = []
        for i in range(self.num_features):
            # (B, C, H, W) * (H, W) -> (B, C)
            feat = (x * weights[i].unsqueeze(0).unsqueeze(0)).sum(dim=(-2, -1))
            features.append(feat)
        
        # Concatenate
        return torch.cat(features, dim=-1)  # (B, C * num_features)


class RewardClassifier(nn.Module):
    """
    基于预训练 ResNet 的奖励分类器（参考 HIL-SERL）
    
    架构：
    - 预训练 ResNet encoder（冻结）
    - Spatial Learned Embeddings pooling
    - MLP 分类头
    
    多相机：每个相机独立编码，然后拼接特征
    """
    
    def __init__(
        self,
        num_cameras: int = 2,
        encoder_name: str = "resnet18",
        freeze_encoder: bool = True,
        pretrained: bool = True,
        hidden_dim: int = 256,
        num_spatial_blocks: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_cameras = num_cameras
        
        # 共享的预训练编码器
        self.encoder = PretrainedResNetEncoder(
            model_name=encoder_name,
            freeze=freeze_encoder,
            pretrained=pretrained,
        )
        
        # 每个相机的 Spatial Learned Embeddings（不共享）
        self.spatial_embeddings = nn.ModuleList([
            SpatialLearnedEmbeddings(
                height=self.encoder.out_size,
                width=self.encoder.out_size,
                channels=self.encoder.out_channels,
                num_features=num_spatial_blocks,
            )
            for _ in range(num_cameras)
        ])
        
        # 特征维度
        feature_dim = self.encoder.out_channels * num_spatial_blocks * num_cameras
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
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
        for i, img in enumerate(images):
            # 编码
            feat = self.encoder(img)  # (B, C', H', W')
            # Spatial pooling
            feat = self.spatial_embeddings[i](feat)  # (B, C' * num_blocks)
            features.append(feat)
        
        # 拼接所有相机特征
        combined = torch.cat(features, dim=-1)
        logits = self.classifier(combined)
        return logits
    
    def predict_reward(
        self,
        images: List[torch.Tensor],
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """预测奖励（二值化）"""
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
    
    @classmethod
    def load_pretrained(cls, path: str, config: Dict = None) -> "RewardClassifier":
        """加载预训练模型"""
        checkpoint = torch.load(path, map_location="cpu")
        
        # 从 checkpoint 获取配置
        saved_config = checkpoint.get("config", {})
        if config:
            saved_config.update(config)
        
        model = cls(
            num_cameras=saved_config.get("num_cameras", 2),
            encoder_name=saved_config.get("encoder", "resnet18"),
            freeze_encoder=True,  # 推理时总是冻结
            pretrained=False,  # 不需要重新加载预训练权重
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model


# ============ 数据增强（参考 HIL-SERL）============

def get_train_transforms(image_size: int = 224, crop_padding: int = 4):
    """训练时的数据增强"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size + crop_padding * 2, image_size + crop_padding * 2)),
        transforms.RandomCrop(image_size),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225],   # ImageNet std
        ),
    ])


def get_eval_transforms(image_size: int = 224):
    """评估时的数据变换（无增强）"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class PickleClassifierDataset(Dataset):
    """
    从 pickle 文件加载的 Classifier 数据集（HIL-SERL 格式）
    
    支持两种初始化方式：
    1. 指定文件路径: success_path, failure_path
    2. 指定数据列表: success_data, failure_data
    """
    
    def __init__(
        self,
        camera_keys: List[str],
        image_size: int = 224,
        balance: bool = True,
        augment: bool = True,
        # 方式 1: 文件路径
        success_path: Optional[str] = None,
        failure_path: Optional[str] = None,
        # 方式 2: 数据列表（支持多文件合并）
        success_data: Optional[List[Dict]] = None,
        failure_data: Optional[List[Dict]] = None,
    ):
        self.camera_keys = camera_keys
        self.image_size = image_size
        self.samples = []  # List of (images_dict, label)
        
        # 数据增强
        self.transform = get_train_transforms(image_size) if augment else get_eval_transforms(image_size)
        
        # 加载成功样本
        if success_data is not None:
            successes = success_data
        elif success_path is not None:
            with open(success_path, "rb") as f:
                successes = pkl.load(f)
        else:
            raise ValueError("Must provide either success_path or success_data")
        
        for t in successes:
            # observations 是 images dict 或包含 images 的 dict
            obs = t.get("observations", t.get("next_observations", {}))
            images = self._extract_images(obs)
            if images:
                self.samples.append((images, 1))
        
        num_successes = len(self.samples)
        
        # 加载失败样本
        if failure_data is not None:
            failures = failure_data
        elif failure_path is not None:
            with open(failure_path, "rb") as f:
                failures = pkl.load(f)
        else:
            raise ValueError("Must provide either failure_path or failure_data")
        
        # 平衡采样（可选）
        if balance and len(failures) > num_successes:
            indices = np.random.choice(len(failures), size=num_successes, replace=False)
            failures = [failures[i] for i in indices]
        
        for t in failures:
            obs = t.get("observations", t.get("next_observations", {}))
            images = self._extract_images(obs)
            if images:
                self.samples.append((images, 0))
        
        print(f"[Dataset] Loaded {len(self.samples)} samples")
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
        # 确保是 HWC 格式和 uint8
        if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
            img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
        
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        # 确保是 RGB 3 通道
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[-1] == 1:
            img = np.concatenate([img, img, img], axis=-1)
        elif img.shape[-1] == 4:
            img = img[:, :, :3]
        
        return self.transform(img)
    
    def set_augment(self, augment: bool):
        """切换是否使用数据增强"""
        self.transform = get_train_transforms(self.image_size) if augment else get_eval_transforms(self.image_size)


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
    parser = argparse.ArgumentParser(description="Train Reward Classifier (HIL-SERL style)")
    
    # 数据源（二选一）
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data_dir", type=str,
                            help="Data directory (auto-match success/failure files)")
    data_group.add_argument("--success_data", type=str, 
                            help="Success pickle path (requires --failure_data)")
    
    parser.add_argument("--failure_data", type=str, 
                        help="Failure pickle path (required with --success_data)")
    parser.add_argument("--output", type=str, default="checkpoints/classifier.pt",
                        help="Output model path")
    parser.add_argument("--cameras", type=str, nargs="+", 
                        default=["cam_high", "cam_left_wrist", "cam_right_wrist"],
                        help="Camera names to use")
    parser.add_argument("--balance", action="store_true", default=True,
                        help="Balance positive and negative samples")
    
    # 模型参数
    parser.add_argument("--encoder", type=str, default="resnet18",
                        choices=["resnet10", "resnet18", "resnet34", "resnet50"],
                        help="Pretrained encoder (default: resnet18)")
    parser.add_argument("--freeze_encoder", action="store_true", default=True,
                        help="Freeze pretrained encoder (HIL-SERL default)")
    parser.add_argument("--no_freeze_encoder", dest="freeze_encoder", action="store_false",
                        help="Train encoder end-to-end")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Classifier hidden dimension")
    parser.add_argument("--num_spatial_blocks", type=int, default=8,
                        help="Number of spatial learned embeddings")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--image_size", type=int, default=224, 
                        help="Image size (default: 224 for ImageNet pretrained)")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split")
    parser.add_argument("--no_augment", action="store_true", help="Disable data augmentation")
    args = parser.parse_args()
    
    # 校验参数
    if args.success_data and not args.failure_data:
        parser.error("--failure_data is required when using --success_data")
    
    logger = Logger(log_dir="./logs")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    logger.info("=" * 60)
    logger.info("Reward Classifier Training (HIL-SERL style)")
    logger.info("=" * 60)
    logger.info(f"Encoder: {args.encoder} (freeze={args.freeze_encoder})")
    logger.info(f"Cameras: {args.cameras}")
    logger.info(f"Image size: {args.image_size}")
    logger.info(f"Device: {device}") 
    
    # 创建数据集
    if args.data_dir:
        # 自动匹配目录中的 success/failure 文件
        logger.info(f"Auto-matching files in: {args.data_dir}")
        
        success_files, failure_files = find_classifier_data(args.data_dir)
        logger.info(f"Found {len(success_files)} success files, {len(failure_files)} failure files")
        
        if not success_files or not failure_files:
            logger.error("Need at least one success and one failure file")
            return
        
        success_data = load_pickle_data(success_files)
        failure_data = load_pickle_data(failure_files)
        logger.info(f"Loaded {len(success_data)} success samples, {len(failure_data)} failure samples")
        
        dataset = PickleClassifierDataset(
            camera_keys=args.cameras,
            success_data=success_data,
            failure_data=failure_data,
            image_size=args.image_size,
            balance=args.balance,
            augment=not args.no_augment,
        )
    else:
        # HIL-SERL pickle 格式（手动指定文件）
        logger.info(f"Loading from pickle: {args.success_data}, {args.failure_data}")
        
        dataset = PickleClassifierDataset(
            success_path=args.success_data,
            failure_path=args.failure_data,
            camera_keys=args.cameras,
            image_size=args.image_size,
            balance=args.balance,
            augment=not args.no_augment,
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
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    ) if val_size > 0 else None
    
    # 创建模型（使用预训练 ResNet）
    model = RewardClassifier(
        num_cameras=len(args.cameras),
        encoder_name=args.encoder,
        freeze_encoder=args.freeze_encoder,
        pretrained=True,
        hidden_dim=args.hidden_dim,
        num_spatial_blocks=args.num_spatial_blocks,
    ).to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params/1e6:.2f}M params, {trainable_params/1e6:.2f}M trainable")
    
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
            "encoder": args.encoder,
            "hidden_dim": args.hidden_dim,
            "num_spatial_blocks": args.num_spatial_blocks,
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
