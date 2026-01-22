"""干预数据Buffer"""
from typing import Dict, Any
from .replay_buffer import ReplayBuffer
from core.orchestration import register_buffer


@register_buffer("intervention")
class InterventionBuffer(ReplayBuffer):
    """干预数据专用Buffer"""
    
    def __init__(self, capacity: int):
        super().__init__(capacity)
    
    def add(self, data: Dict[str, Any]) -> None:
        data["source"] = "intervention"
        super().add(data)
