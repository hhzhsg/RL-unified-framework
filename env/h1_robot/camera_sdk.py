"""
相机图像采集模块

通过 gRPC 从相机服务获取图像（WebRTC + gRPC 混合协议）。
支持双目相机分割：每张图切成 _0 和 _1 两个视角。

依赖:
    - lib/camera_client.py
    - lib/robot_pb2.py
"""
import time
import logging
import threading
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class ImageRecorder:
    """
    相机图像采集器
    
    使用方式:
        recorder = ImageRecorder(
            camera_names=['v4l2/cam_high', 'v4l2/cam_right_wrist'],
            grpc_target='localhost:50051',
            split_stereo=True,
        )
        images = recorder.get_images()
        # images = {
        #     'cam_high_0': np.ndarray,
        #     'cam_high_1': np.ndarray,
        #     'cam_right_wrist_0': np.ndarray,
        #     'cam_right_wrist_1': np.ndarray,
        # }
    """
    
    def __init__(
        self,
        camera_names: List[str],
        grpc_target: str = "localhost:50051",
        split_stereo: bool = True,
    ):
        """
        Args:
            camera_names: 相机名称列表，如 ['v4l2/cam_high', 'v4l2/cam_right_wrist']
            grpc_target: 相机 gRPC 服务地址
            split_stereo: 是否将双目图像分割成左右两半
        """
        self.camera_names = camera_names
        self.grpc_target = grpc_target
        self.split_stereo = split_stereo
        
        # 导入 lib 目录下的 client
        from .lib.camera_client import UnifiedReceiverClient
        
        self.cli = UnifiedReceiverClient(grpc_target=grpc_target)
        self.stop_event = threading.Event()
        self.image_dict: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        
        # 启动客户端
        try:
            self.cli.start()
            logger.info(f"[ImageRecorder] Connected to {grpc_target}")
        except Exception as e:
            logger.error(f"[ImageRecorder] Failed to connect: {e}")
            raise
        
        # 为每个相机启动采集线程
        for cam_name in camera_names:
            t = threading.Thread(target=self._handler, args=(cam_name,), daemon=True)
            t.start()
    
    def _handler(self, cam_name: str) -> None:
        """相机采集线程"""
        # 从 v4l2/cam_high 提取纯名称 cam_high
        pure_name = cam_name.replace('v4l2/', '').replace('rs/', '')
        
        while not self.stop_event.is_set():
            try:
                item = self.cli.get_latest_frame(cam_name)
                if item is None:
                    time.sleep(0.01)
                    continue
                
                bgr, ts = item
                
                with self._lock:
                    if self.split_stereo:
                        # 双目分割：左右各一半
                        h, w = bgr.shape[:2]
                        mid = w // 2
                        self.image_dict[f"{pure_name}_0"] = bgr[:, :mid].copy()
                        self.image_dict[f"{pure_name}_1"] = bgr[:, mid:].copy()
                    else:
                        self.image_dict[pure_name] = bgr.copy()
                        
            except Exception as e:
                logger.warning(f"[ImageRecorder] Error on {cam_name}: {e}")
                time.sleep(0.1)
    
    def get_images(self) -> Dict[str, np.ndarray]:
        """获取当前所有相机图像"""
        with self._lock:
            return self.image_dict.copy()
    
    def get_image(self, cam_name: str) -> Optional[np.ndarray]:
        """获取单个相机图像"""
        with self._lock:
            return self.image_dict.get(cam_name)
    
    def stop(self) -> None:
        """停止采集"""
        if hasattr(self, 'stop_event'):
            self.stop_event.set()
        if hasattr(self, 'cli') and hasattr(self.cli, 'stop'):
            self.cli.stop()
        logger.info("[ImageRecorder] Stopped")
    
    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass  # 忽略析构时的错误


if __name__ == '__main__':
    # 测试
    logging.basicConfig(level=logging.INFO)
    recorder = ImageRecorder(
        camera_names=['v4l2/cam_high', 'v4l2/cam_right_wrist'],
        grpc_target='localhost:50051',
    )
    
    while True:
        images = recorder.get_images()
        print(f"Got {len(images)} images: {list(images.keys())}")
        for name, img in images.items():
            print(f"  {name}: shape={img.shape}")
        time.sleep(1)
