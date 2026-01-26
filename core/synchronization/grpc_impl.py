"""
gRPC Actor-Learner 通信实现

基于 protobuf 的分布式通信，支持跨机器训练。
Actor 运行在机器人控制电脑，Learner 运行在 GPU 服务器。

依赖:
    pip install grpcio grpcio-tools

生成 protobuf 代码:
    cd core/synchronization/protos
    python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. actor_learner.proto
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import threading
import time
import io
import pickle
import numpy as np
import torch

try:
    import grpc
    from concurrent import futures
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False
    print("[Warning] grpcio not installed. Install with: pip install grpcio grpcio-tools")

from .actor_learner import (
    ActorLearnerConfig,
    LearnerServerInterface,
    ActorClientInterface,
)


# ============ 序列化工具 ============

def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """将 Tensor 序列化为 bytes"""
    buffer = io.BytesIO()
    # 使用 numpy 序列化（更通用）
    np_array = tensor.cpu().numpy()
    np.save(buffer, np_array, allow_pickle=False)
    return buffer.getvalue()


def bytes_to_tensor(data: bytes, device: str = "cpu") -> torch.Tensor:
    """将 bytes 反序列化为 Tensor"""
    buffer = io.BytesIO(data)
    np_array = np.load(buffer, allow_pickle=False)
    return torch.from_numpy(np_array).to(device)


def serialize_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, bytes]:
    """序列化整个 state_dict"""
    return {k: tensor_to_bytes(v) for k, v in state_dict.items()}


def deserialize_state_dict(data: Dict[str, bytes], device: str = "cpu") -> Dict[str, torch.Tensor]:
    """反序列化整个 state_dict"""
    return {k: bytes_to_tensor(v, device) for k, v in data.items()}


def serialize_transition(transition: Dict[str, Any]) -> Dict[str, Any]:
    """序列化单个 transition"""
    result = {}
    for k, v in transition.items():
        if isinstance(v, np.ndarray):
            result[k] = v.tobytes()
            result[f"__{k}_shape"] = v.shape
            result[f"__{k}_dtype"] = str(v.dtype)
        elif isinstance(v, (dict, list)):
            result[k] = pickle.dumps(v)
            result[f"__{k}_pickle"] = True
        else:
            result[k] = v
    return result


def deserialize_transition(data: Dict[str, Any]) -> Dict[str, Any]:
    """反序列化单个 transition"""
    result = {}
    processed_keys = set()
    
    for k, v in data.items():
        if k.startswith("__"):
            continue
        
        if f"__{k}_shape" in data:
            # numpy array
            shape = data[f"__{k}_shape"]
            dtype = data[f"__{k}_dtype"]
            result[k] = np.frombuffer(v, dtype=dtype).reshape(shape)
            processed_keys.add(k)
        elif f"__{k}_pickle" in data:
            # pickled object
            result[k] = pickle.loads(v)
            processed_keys.add(k)
        elif k not in processed_keys:
            result[k] = v
    
    return result


# ============ gRPC Servicer（服务端逻辑） ============

class ActorLearnerServicer:
    """gRPC 服务端实现"""
    
    def __init__(self):
        self._transition_queue: List[Dict] = []
        self._transition_lock = threading.Lock()
        
        self._weights: Optional[Dict[str, bytes]] = None
        self._weights_version = 0
        self._weights_lock = threading.Lock()
        self._train_step = 0
    
    def SendTransitions(self, request, context):
        """接收 transitions"""
        try:
            with self._transition_lock:
                for t_proto in request.transitions:
                    transition = {
                        "obs": pickle.loads(t_proto.obs) if t_proto.obs else None,
                        "action": np.frombuffer(t_proto.action, dtype=np.float32) if t_proto.action else None,
                        "policy_action": np.frombuffer(t_proto.policy_action, dtype=np.float32) if t_proto.policy_action else None,
                        "reward": t_proto.reward,
                        "next_obs": pickle.loads(t_proto.next_obs) if t_proto.next_obs else None,
                        "done": t_proto.done,
                        "is_intervention": t_proto.is_intervention,
                        "source": t_proto.source,
                    }
                    self._transition_queue.append(transition)
            
            # 返回成功响应
            return self._create_status_response(True, f"Received {len(request.transitions)} transitions")
        except Exception as e:
            return self._create_status_response(False, str(e))
    
    def GetWeights(self, request, context):
        """返回最新权重"""
        with self._weights_lock:
            if self._weights is None:
                # 返回空权重
                return self._create_weights_proto({}, 0, 0)
            
            # 检查是否需要更新
            if request.last_version >= self._weights_version:
                return self._create_weights_proto({}, self._weights_version, self._train_step)
            
            return self._create_weights_proto(
                self._weights, 
                self._weights_version, 
                self._train_step
            )
    
    def CheckWeightsUpdate(self, request, context):
        """检查是否有权重更新"""
        with self._weights_lock:
            has_update = self._weights_version > request.last_version
            return self._create_status_response(
                has_update, 
                f"version: {self._weights_version}"
            )
    
    def HealthCheck(self, request, context):
        """健康检查"""
        return self._create_status_response(True, "OK")
    
    # === 内部方法（供 GRPCLearnerServer 调用）===
    
    def get_transitions(self) -> List[Dict]:
        """获取并清空 transition 队列"""
        with self._transition_lock:
            transitions = self._transition_queue.copy()
            self._transition_queue.clear()
            return transitions
    
    def publish_weights(self, state_dict: Dict[str, torch.Tensor], train_step: int = 0):
        """发布新权重"""
        serialized = serialize_state_dict(state_dict)
        with self._weights_lock:
            self._weights = serialized
            self._weights_version += 1
            self._train_step = train_step
    
    def _create_status_response(self, success: bool, message: str):
        """创建状态响应（需要导入 pb2）"""
        # 延迟导入
        from .protos import actor_learner_pb2
        return actor_learner_pb2.StatusResponse(success=success, message=message)
    
    def _create_weights_proto(self, weights: Dict[str, bytes], version: int, train_step: int):
        """创建权重响应"""
        from .protos import actor_learner_pb2
        
        response = actor_learner_pb2.WeightsProto(
            version=version,
            train_step=train_step,
            timestamp=int(time.time() * 1000),
        )
        
        for name, data in weights.items():
            tensor_proto = actor_learner_pb2.TensorProto(data=data)
            response.state_dict[name].CopyFrom(tensor_proto)
        
        return response


# ============ gRPC Server ============

class GRPCLearnerServer(LearnerServerInterface):
    """
    gRPC Learner 服务器
    
    在 GPU 服务器上运行，接收 Actor 的 transitions，发布权重更新。
    
    使用示例:
        server = GRPCLearnerServer(config)
        server.start()
        
        # 训练循环中
        transitions = server.recv_transitions()
        # ... 训练 ...
        server.publish_weights(model.state_dict())
        
        server.stop()
    """
    
    def __init__(self, config: Optional[ActorLearnerConfig] = None):
        if not GRPC_AVAILABLE:
            raise ImportError("grpcio is required. Install with: pip install grpcio grpcio-tools")
        
        self.config = config or ActorLearnerConfig()
        self._server: Optional[grpc.Server] = None
        self._servicer: Optional[ActorLearnerServicer] = None
        self._running = False
    
    def start(self) -> None:
        """启动 gRPC 服务器"""
        from .protos import actor_learner_pb2_grpc
        
        self._servicer = ActorLearnerServicer()
        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=[
                ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
                ('grpc.max_receive_message_length', 100 * 1024 * 1024),
            ]
        )
        
        actor_learner_pb2_grpc.add_ActorLearnerServiceServicer_to_server(
            self._servicer, self._server
        )
        
        address = f"0.0.0.0:{self.config.learner_port}"
        self._server.add_insecure_port(address)
        self._server.start()
        self._running = True
        
        print(f"[gRPC-Learner] Server started on {address}")
    
    def stop(self) -> None:
        """停止服务器"""
        if self._server:
            self._server.stop(grace=5)
            self._running = False
            print("[gRPC-Learner] Server stopped")
    
    def recv_transitions(self, block: bool = False, timeout: float = 0.1) -> List[Dict[str, Any]]:
        """接收 transitions"""
        if not self._servicer:
            return []
        
        if block:
            # 阻塞等待
            start_time = time.time()
            while time.time() - start_time < timeout:
                transitions = self._servicer.get_transitions()
                if transitions:
                    return transitions
                time.sleep(0.01)
            return []
        else:
            return self._servicer.get_transitions()
    
    def publish_weights(self, state_dict: Dict[str, torch.Tensor], metadata: Optional[Dict] = None) -> None:
        """发布权重"""
        if self._servicer:
            train_step = metadata.get("step", 0) if metadata else 0
            self._servicer.publish_weights(state_dict, train_step)


# ============ gRPC Client ============

class GRPCActorClient(ActorClientInterface):
    """
    gRPC Actor 客户端
    
    在机器人控制电脑上运行，发送 transitions 到 Learner，接收权重更新。
    
    使用示例:
        client = GRPCActorClient(config)
        client.connect()
        
        # Actor 循环中
        client.send_transitions(transitions)
        weights = client.recv_weights(block=False)
        if weights:
            policy.load_state_dict(weights)
        
        client.disconnect()
    """
    
    def __init__(self, config: Optional[ActorLearnerConfig] = None):
        if not GRPC_AVAILABLE:
            raise ImportError("grpcio is required. Install with: pip install grpcio grpcio-tools")
        
        self.config = config or ActorLearnerConfig()
        self._channel: Optional[grpc.Channel] = None
        self._stub = None
        self._connected = False
        self._last_weight_version = -1
    
    def connect(self) -> bool:
        """连接到 Learner 服务器"""
        from .protos import actor_learner_pb2_grpc, actor_learner_pb2
        
        address = f"{self.config.learner_host}:{self.config.learner_port}"
        
        try:
            self._channel = grpc.insecure_channel(
                address,
                options=[
                    ('grpc.max_send_message_length', 100 * 1024 * 1024),
                    ('grpc.max_receive_message_length', 100 * 1024 * 1024),
                ]
            )
            self._stub = actor_learner_pb2_grpc.ActorLearnerServiceStub(self._channel)
            
            # 健康检查
            response = self._stub.HealthCheck(actor_learner_pb2.Empty())
            if response.success:
                self._connected = True
                print(f"[gRPC-Actor] Connected to {address}")
                return True
            else:
                print(f"[gRPC-Actor] Health check failed: {response.message}")
                return False
                
        except grpc.RpcError as e:
            print(f"[gRPC-Actor] Connection failed: {e}")
            return False
    
    def disconnect(self) -> None:
        """断开连接"""
        if self._channel:
            self._channel.close()
            self._connected = False
            print("[gRPC-Actor] Disconnected")
    
    def send_transition(self, transition: Dict[str, Any]) -> None:
        """发送单个 transition"""
        self.send_transitions([transition])
    
    def send_transitions(self, transitions: List[Dict[str, Any]]) -> None:
        """批量发送 transitions"""
        if not self._connected or not self._stub:
            raise RuntimeError("Not connected to learner server")
        
        from .protos import actor_learner_pb2
        
        # 构建请求
        batch = actor_learner_pb2.TransitionBatch(
            timestamp=int(time.time() * 1000)
        )
        
        for t in transitions:
            t_proto = actor_learner_pb2.TransitionProto(
                obs=pickle.dumps(t.get("obs")),
                action=np.asarray(t.get("action", []), dtype=np.float32).tobytes(),
                policy_action=np.asarray(t.get("policy_action", []), dtype=np.float32).tobytes(),
                reward=float(t.get("reward", 0)),
                next_obs=pickle.dumps(t.get("next_obs")),
                done=bool(t.get("done", False)),
                is_intervention=bool(t.get("is_intervention", False)),
                source=str(t.get("source", "rollout")),
            )
            batch.transitions.append(t_proto)
        
        try:
            response = self._stub.SendTransitions(batch)
            if not response.success:
                print(f"[gRPC-Actor] Send failed: {response.message}")
        except grpc.RpcError as e:
            print(f"[gRPC-Actor] RPC error: {e}")
    
    def recv_weights(self, block: bool = True, timeout: float = 10.0) -> Optional[Dict[str, torch.Tensor]]:
        """接收最新权重"""
        if not self._connected or not self._stub:
            raise RuntimeError("Not connected to learner server")
        
        from .protos import actor_learner_pb2
        
        try:
            if block:
                # 阻塞轮询
                start_time = time.time()
                while time.time() - start_time < timeout:
                    response = self._stub.GetWeights(
                        actor_learner_pb2.WeightRequest(last_version=self._last_weight_version)
                    )
                    
                    if response.version > self._last_weight_version and response.state_dict:
                        self._last_weight_version = response.version
                        return self._deserialize_weights(response)
                    
                    time.sleep(self.config.weight_poll_interval)
                return None
            else:
                # 非阻塞
                response = self._stub.GetWeights(
                    actor_learner_pb2.WeightRequest(last_version=self._last_weight_version)
                )
                
                if response.version > self._last_weight_version and response.state_dict:
                    self._last_weight_version = response.version
                    return self._deserialize_weights(response)
                return None
                
        except grpc.RpcError as e:
            print(f"[gRPC-Actor] RPC error: {e}")
            return None
    
    def _deserialize_weights(self, response) -> Dict[str, torch.Tensor]:
        """反序列化权重"""
        result = {}
        for name, tensor_proto in response.state_dict.items():
            result[name] = bytes_to_tensor(tensor_proto.data)
        return result


__all__ = [
    "GRPCLearnerServer",
    "GRPCActorClient",
    "tensor_to_bytes",
    "bytes_to_tensor",
    "serialize_state_dict",
    "deserialize_state_dict",
]
