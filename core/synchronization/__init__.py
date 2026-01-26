"""
Synchronization模块

训练/推理同步
"""
from .base_sync import BaseSynchronizer
from .weight_publisher import WeightPublisher
from .weight_subscriber import WeightSubscriber
from .checkpoint_manager import CheckpointManager

from .local import SharedMemorySync, QueueSync

# Actor-Learner 通信层（推荐使用）
from .actor_learner import (
    ActorLearnerConfig,
    LearnerServerInterface,
    ActorClientInterface,
    LocalLearnerServer,
    LocalActorClient,
    create_learner_server,
    create_actor_client,
)

# gRPC 实现（懒加载，避免未安装 grpcio 时报错）
def get_grpc_classes():
    """获取 gRPC 实现类"""
    from .grpc_impl import GRPCLearnerServer, GRPCActorClient
    return GRPCLearnerServer, GRPCActorClient

__all__ = [
    # 基础同步（旧 API，兼容性保留）
    "BaseSynchronizer",
    "WeightPublisher",
    "WeightSubscriber",
    "CheckpointManager",
    # Local
    "SharedMemorySync",
    "QueueSync",
    # Actor-Learner（推荐使用）
    "ActorLearnerConfig",
    "LearnerServerInterface",
    "ActorClientInterface",
    "LocalLearnerServer",
    "LocalActorClient",
    "get_grpc_classes",  # 懒加载 gRPC 实现
    "create_learner_server",
    "create_actor_client",
]
