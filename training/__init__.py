"""Training-side policy model and observation utilities."""

import os

# Policy groups and history prefixes vary throughout collection. Expandable
# segments let the native allocator reuse those neighboring attention buffers
# instead of retaining a fragmented segment for every observed shape.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from .model import (
    BloodFlowTransformer,
    HistoryEncoder,
    PolicyOutput,
    TransformerConfig,
)
from .observation import unpack_action_masks

__all__ = [
    "BloodFlowTransformer",
    "HistoryEncoder",
    "PolicyOutput",
    "TransformerConfig",
    "unpack_action_masks",
]
