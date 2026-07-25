"""Training-side models and buffer utilities for Blood Flow Mahjong."""

import os

# Set this before importing torch/model code so CUDA uses expandable allocator
# segments for the variable-sized rollout KV cache.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from .model import (
    ActorCriticOutput,
    BloodFlowTransformer,
    HistoryEncoder,
    HistoryKVCache,
    TransformerConfig,
)
from .observation import unpack_action_masks

__all__ = [
    "ActorCriticOutput",
    "BloodFlowTransformer",
    "HistoryEncoder",
    "HistoryKVCache",
    "TransformerConfig",
    "unpack_action_masks",
]
