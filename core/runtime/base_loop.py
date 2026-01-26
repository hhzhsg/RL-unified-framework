"""
运行循环基类

所有 Loop 的基类，提供:
- step() 抽象方法
- run() 循环执行
- setup() / cleanup() 生命周期钩子
- 停止机制（stop_event）
- 回调机制（callbacks）
- 步计数（step_count）

Loop 继承关系（Actor-Learner-Evaluator 架构）:
  BaseLoop
  ├── ActorLoop         # Online RL 环境交互
  ├── LearnerLoop       # Offline/Online RL 训练
  ├── EvaluatorLoop     # 策略评估
  ├── HILActorLoop      # HIL Actor（独立进程）
  └── HILLearnerLoop    # HIL Learner（独立进程）

场景选择:
  - Offline RL: LearnerLoop
  - Online RL: ActorLoop + LearnerLoop
  - HIL: HILActorLoop + HILLearnerLoop（两个独立进程）
  - 评估: EvaluatorLoop
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
import time
import threading


class BaseLoop(ABC):
    """运行循环基类"""
    
    def __init__(self):
        self._is_running = False
        self._step_count = 0
        self._callbacks: Dict[str, Callable] = {}
        self._stop_event = threading.Event()
    
    @abstractmethod
    def step(self) -> Dict[str, Any]:
        """
        执行单步
        
        Returns:
            step_info: 单步信息
        """
        pass
    
    def run(self, num_steps: int, log_freq: int = 100) -> Dict[str, Any]:
        """
        运行循环
        
        Args:
            num_steps: 运行步数
            log_freq: 日志频率
            
        Returns:
            运行结果统计
        """
        self._is_running = True
        self._stop_event.clear()
        
        total_metrics = {}
        start_time = time.time()
        
        for _ in range(num_steps):
            if self._stop_event.is_set():
                break
            
            # 执行单步
            step_info = self.step()
            # 如果子类没有手动管理步计数，才自动递增
            if not getattr(self, "_manual_step_increment", False):
                self._step_count += 1
            
            # 累积metrics
            for k, v in step_info.items():
                if isinstance(v, (int, float)):
                    if k not in total_metrics:
                        total_metrics[k] = 0.0
                    total_metrics[k] += v
            
            # 日志
            if self._step_count % log_freq == 0:
                self._log(step_info)
            
            # 回调
            self._trigger_callbacks("on_step", step_info)
        
        self._is_running = False
        
        elapsed = time.time() - start_time
        return {
            "total_steps": self._step_count,
            "elapsed_time": elapsed,
            "steps_per_second": num_steps / elapsed if elapsed > 0 else 0,
            **{k: v / num_steps for k, v in total_metrics.items()},
        }
    
    def stop(self) -> None:
        """停止循环"""
        self._stop_event.set()
    
    @property
    def is_running(self) -> bool:
        return self._is_running
    
    @property
    def step_count(self) -> int:
        return self._step_count
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """注册回调"""
        self._callbacks[event] = callback
    
    def _trigger_callbacks(self, event: str, data: Any) -> None:
        """触发回调"""
        if event in self._callbacks:
            self._callbacks[event](data)
    
    def _log(self, info: Dict[str, Any]) -> None:
        """打印日志"""
        metrics_str = ", ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                 for k, v in info.items())
        print(f"[Step {self._step_count}] {metrics_str}")
