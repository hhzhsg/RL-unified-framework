"""
Actor-Learner 同步通信层

基于 HIL-SERL 的分布式架构设计
支持 Local (调试) 和 gRPC (分布式) 两种模式
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from queue import Queue, Empty
import threading
import torch


@dataclass
class ActorLearnerConfig:
    """Actor-Learner 配置"""
    learner_host: str = "localhost"
    learner_port: int = 50051
    weight_poll_interval: float = 0.1  # 秒
    policy_push_frequency: int = 100   # 训练步数
    transition_batch_size: int = 10    # 每次发送的transition数量


# ============ 抽象接口 ============

class LearnerServerInterface(ABC):
    """Learner端服务器接口"""
    
    @abstractmethod
    def start(self) -> None:
        """启动服务器"""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """停止服务器"""
        pass
    
    @abstractmethod
    def recv_transitions(self, block: bool = False, timeout: float = 0.1) -> List[Dict[str, Any]]:
        """
        接收来自Actor的transitions
        
        Args:
            block: 是否阻塞等待
            timeout: 超时时间（秒）
        Returns:
            transition列表
        """
        pass
    
    @abstractmethod
    def publish_weights(self, state_dict: Dict[str, torch.Tensor], metadata: Optional[Dict] = None) -> None:
        """
        发布最新策略权重
        
        Args:
            state_dict: 模型状态字典
            metadata: 元数据（训练步数等）
        """
        pass


class ActorClientInterface(ABC):
    """Actor端客户端接口"""
    
    @abstractmethod
    def connect(self) -> bool:
        """连接到Learner服务器"""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """断开连接"""
        pass
    
    @abstractmethod
    def send_transition(self, transition: Dict[str, Any]) -> None:
        """发送单个transition到Learner"""
        pass
    
    @abstractmethod
    def send_transitions(self, transitions: List[Dict[str, Any]]) -> None:
        """批量发送transitions"""
        pass
    
    @abstractmethod
    def recv_weights(self, block: bool = True, timeout: float = 10.0) -> Optional[Dict[str, torch.Tensor]]:
        """
        接收最新策略权重
        
        Args:
            block: 是否阻塞等待
            timeout: 超时时间（秒）
        Returns:
            模型状态字典，无新权重时返回None
        """
        pass


# ============ Local实现（用于调试，同进程） ============

class LocalLearnerServer(LearnerServerInterface):
    """
    本地Learner服务器（基于Queue，用于调试）
    
    适用于单进程/多线程场景
    """
    
    def __init__(self, config: Optional[ActorLearnerConfig] = None):
        self.config = config or ActorLearnerConfig()
        self._transition_queue: Queue = Queue()
        self._weight_queue: Queue = Queue(maxsize=1)
        self._running = False
        self._lock = threading.Lock()
        self._weight_version = 0
    
    def start(self) -> None:
        self._running = True
    
    def stop(self) -> None:
        self._running = False
    
    def recv_transitions(self, block: bool = False, timeout: float = 0.1) -> List[Dict[str, Any]]:
        transitions = []
        try:
            if block:
                t = self._transition_queue.get(timeout=timeout)
                transitions.append(t)
            
            # 尽可能多地取出队列中的数据
            while True:
                try:
                    t = self._transition_queue.get_nowait()
                    transitions.append(t)
                except Empty:
                    break
        except Empty:
            pass
        return transitions
    
    def publish_weights(self, state_dict: Dict[str, torch.Tensor], metadata: Optional[Dict] = None) -> None:
        with self._lock:
            self._weight_version += 1
            # 清空旧权重
            while not self._weight_queue.empty():
                try:
                    self._weight_queue.get_nowait()
                except Empty:
                    break
            
            # CPU序列化
            cpu_state_dict = {k: v.cpu().clone() for k, v in state_dict.items()}
            self._weight_queue.put({
                "state_dict": cpu_state_dict,
                "metadata": metadata or {},
                "version": self._weight_version,
            })
    
    # 暴露给LocalActorClient使用
    def _get_transition_queue(self) -> Queue:
        return self._transition_queue
    
    def _get_weight_queue(self) -> Queue:
        return self._weight_queue


class LocalActorClient(ActorClientInterface):
    """
    本地Actor客户端（直接引用Server的Queue）
    
    适用于单进程/多线程场景
    """
    
    def __init__(self, server: LocalLearnerServer):
        self._server = server
        self._connected = False
        self._last_weight_version = -1
    
    def connect(self) -> bool:
        self._connected = True
        return True
    
    def disconnect(self) -> None:
        self._connected = False
    
    def send_transition(self, transition: Dict[str, Any]) -> None:
        if not self._connected:
            raise RuntimeError("Not connected to learner server")
        self._server._get_transition_queue().put(transition)
    
    def send_transitions(self, transitions: List[Dict[str, Any]]) -> None:
        for t in transitions:
            self.send_transition(t)
    
    def recv_weights(self, block: bool = True, timeout: float = 10.0) -> Optional[Dict[str, torch.Tensor]]:
        if not self._connected:
            raise RuntimeError("Not connected to learner server")
        
        try:
            if block:
                data = self._server._get_weight_queue().get(timeout=timeout)
            else:
                data = self._server._get_weight_queue().get_nowait()
            
            if data["version"] > self._last_weight_version:
                self._last_weight_version = data["version"]
                return data["state_dict"]
            return None
        except Empty:
            return None


# ============ gRPC实现（用于分布式训练） ============
# 完整实现在 grpc_impl.py，这里提供懒加载包装

def _get_grpc_server_class():
    """懒加载 gRPC Server（避免未安装 grpcio 时报错）"""
    try:
        from .grpc_impl import GRPCLearnerServer
        return GRPCLearnerServer
    except ImportError as e:
        raise ImportError(
            f"gRPC implementation requires grpcio. Install with: pip install grpcio grpcio-tools\n"
            f"Original error: {e}"
        )


def _get_grpc_client_class():
    """懒加载 gRPC Client"""
    try:
        from .grpc_impl import GRPCActorClient
        return GRPCActorClient
    except ImportError as e:
        raise ImportError(
            f"gRPC implementation requires grpcio. Install with: pip install grpcio grpcio-tools\n"
            f"Original error: {e}"
        )


# ============ 工厂函数 ============

def create_learner_server(mode: str = "local", config: Optional[ActorLearnerConfig] = None) -> LearnerServerInterface:
    """创建Learner服务器"""
    if mode == "local":
        return LocalLearnerServer(config)
    elif mode == "grpc":
        GRPCLearnerServer = _get_grpc_server_class()
        return GRPCLearnerServer(config)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def create_actor_client(
    mode: str = "local",
    config: Optional[ActorLearnerConfig] = None,
    server: Optional[LocalLearnerServer] = None,
) -> ActorClientInterface:
    """创建Actor客户端"""
    if mode == "local":
        if server is None:
            raise ValueError("Local mode requires server instance")
        return LocalActorClient(server)
    elif mode == "grpc":
        GRPCActorClient = _get_grpc_client_class()
        return GRPCActorClient(config)
    else:
        raise ValueError(f"Unknown mode: {mode}")


__all__ = [
    "ActorLearnerConfig",
    "LearnerServerInterface",
    "ActorClientInterface",
    "LocalLearnerServer",
    "LocalActorClient",
    "create_learner_server",
    "create_actor_client",
]
