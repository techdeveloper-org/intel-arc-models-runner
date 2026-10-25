"""Flash-Stream heterogeneous inference engine package for heavy models.

Exports the zero-RAM streaming weight manager, heterogeneous layer dispatcher,
bounded circular paged key-value buffer, and autoregressive streaming engine.
"""

from arc_runner.engine.flash_stream.engine import FlashStreamInferenceEngine
from arc_runner.engine.flash_stream.hetero_dispatcher import HeterogeneousLayerDispatcher
from arc_runner.engine.flash_stream.kv_paged_buffer import FlashStreamKVPagedBuffer
from arc_runner.engine.flash_stream.weight_streamer import FlashStreamWeightManager

__all__ = [
    "FlashStreamInferenceEngine",
    "FlashStreamKVPagedBuffer",
    "FlashStreamWeightManager",
    "HeterogeneousLayerDispatcher",
]
