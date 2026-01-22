"""
检查点管理器

管理模型检查点的保存和加载
"""
from typing import Dict, Any, Optional, List
from pathlib import Path
import torch
import json
import shutil


class CheckpointManager:
    """
    检查点管理器
    
    功能:
    - 保存/加载检查点
    - 管理检查点数量
    - 记录最佳检查点
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        metric_name: str = "reward",
        mode: str = "max",  # "max" | "min"
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.mode = mode
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self._checkpoints: List[Dict[str, Any]] = []
        self._best_metric = float("-inf") if mode == "max" else float("inf")
        self._best_checkpoint: Optional[str] = None
        
        # 加载已有检查点记录
        self._load_index()
    
    def save(
        self,
        state_dict: Dict[str, Any],
        step: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        保存检查点
        
        Args:
            state_dict: 状态字典
            step: 训练步数
            metrics: 评估指标
            
        Returns:
            检查点路径
        """
        checkpoint_name = f"checkpoint_step_{step}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # 保存检查点
        torch.save({
            "state_dict": state_dict,
            "step": step,
            "metrics": metrics,
        }, checkpoint_path)
        
        # 记录
        self._checkpoints.append({
            "path": str(checkpoint_path),
            "step": step,
            "metrics": metrics,
        })
        
        # 检查是否是最佳
        if metrics and self.metric_name in metrics:
            metric_value = metrics[self.metric_name]
            is_best = (self.mode == "max" and metric_value > self._best_metric) or \
                      (self.mode == "min" and metric_value < self._best_metric)
            
            if is_best:
                self._best_metric = metric_value
                self._best_checkpoint = str(checkpoint_path)
                
                # 保存best链接
                best_path = self.checkpoint_dir / "best.pt"
                if best_path.exists():
                    best_path.unlink()
                shutil.copy(checkpoint_path, best_path)
        
        # 清理旧检查点
        self._cleanup()
        
        # 保存索引
        self._save_index()
        
        return str(checkpoint_path)
    
    def load(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径，None表示加载最新
            
        Returns:
            检查点内容
        """
        if checkpoint_path is None:
            if self._checkpoints:
                checkpoint_path = self._checkpoints[-1]["path"]
            else:
                raise ValueError("No checkpoint available")
        
        return torch.load(checkpoint_path, map_location="cpu")
    
    def load_best(self) -> Dict[str, Any]:
        """加载最佳检查点"""
        best_path = self.checkpoint_dir / "best.pt"
        if best_path.exists():
            return torch.load(best_path, map_location="cpu")
        raise ValueError("No best checkpoint available")
    
    def _cleanup(self) -> None:
        """清理旧检查点"""
        while len(self._checkpoints) > self.max_checkpoints:
            old = self._checkpoints.pop(0)
            old_path = Path(old["path"])
            if old_path.exists() and str(old_path) != self._best_checkpoint:
                old_path.unlink()
    
    def _save_index(self) -> None:
        """保存索引"""
        index_path = self.checkpoint_dir / "index.json"
        with open(index_path, "w") as f:
            json.dump({
                "checkpoints": self._checkpoints,
                "best_checkpoint": self._best_checkpoint,
                "best_metric": self._best_metric,
            }, f, indent=2)
    
    def _load_index(self) -> None:
        """加载索引"""
        index_path = self.checkpoint_dir / "index.json"
        if index_path.exists():
            with open(index_path, "r") as f:
                data = json.load(f)
                self._checkpoints = data.get("checkpoints", [])
                self._best_checkpoint = data.get("best_checkpoint")
                self._best_metric = data.get("best_metric", 
                    float("-inf") if self.mode == "max" else float("inf"))
    
    @property
    def best_checkpoint(self) -> Optional[str]:
        return self._best_checkpoint
    
    @property
    def latest_checkpoint(self) -> Optional[str]:
        if self._checkpoints:
            return self._checkpoints[-1]["path"]
        return None
