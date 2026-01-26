"""
Protobuf 定义包

包含 gRPC 通信的 protobuf 定义和生成的 Python 代码。

生成代码命令:
    cd core/synchronization/protos
    python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. actor_learner.proto
"""
# 懒加载，避免未安装 grpcio 时报错
def get_pb2_modules():
    """获取生成的 protobuf 模块"""
    from . import actor_learner_pb2, actor_learner_pb2_grpc
    return actor_learner_pb2, actor_learner_pb2_grpc

__all__ = ["get_pb2_modules"]
