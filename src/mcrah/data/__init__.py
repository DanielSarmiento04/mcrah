"""Data modules.

Two paths are supported, matching workflow.md Phase 1:
  * :class:`DNeRFDataset` reads the raw D-NeRF ``transforms_*.json`` + PNGs in
    Python (no preprocessing dependency).
  * :class:`ProcessedDNeRFDataset` consumes the Rust-preprocessed ``.npy``
    tensors for maximum throughput (rules.md Rule 7).

Both paths yield the same :class:`SceneSample` so downstream code is agnostic.
"""

from .scene import DNeRFDataset, ProcessedDNeRFDataset, SceneSample, list_categories
from .collate import collate_scenes, SceneBatch

__all__ = [
    "DNeRFDataset",
    "ProcessedDNeRFDataset",
    "SceneSample",
    "SceneBatch",
    "collate_scenes",
    "list_categories",
]
