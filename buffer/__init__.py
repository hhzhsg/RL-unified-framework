from .base_buffer import BaseBuffer
from .rollout_buffer import RolloutBuffer
from .intervention_buffer import InterventionBuffer
from .data_hub import DataHub  # DataHubV2 renamed to DataHub
from .sample_strategy import (
    BaseSampleStrategy,
    DemoOnlyStrategy,
    RolloutOnlyStrategy,
    MixedStrategy,
    create_strategy,
    STRATEGY_REGISTRY,
)

# HDF5 支持 (可选依赖)
try:
    from .hdf5_buffer import HDF5DemoBuffer, inspect_hdf5
    HAS_HDF5 = True
except ImportError:
    HDF5DemoBuffer = None
    inspect_hdf5 = None
    HAS_HDF5 = False

__all__ = [
    "BaseBuffer",
    "RolloutBuffer",
    "InterventionBuffer",
    "DataHub",
    "BaseSampleStrategy",
    "DemoOnlyStrategy",
    "RolloutOnlyStrategy",
    "MixedStrategy",
    "create_strategy",
    "STRATEGY_REGISTRY",
    # HDF5
    "HDF5DemoBuffer",
    "inspect_hdf5",
    "HAS_HDF5",
]
