"""Collation for variable-length temporal rollouts and multi-view batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from .scene import SceneSample


@dataclass
class SceneBatch:
    images: torch.Tensor      # (B,3,H,W)
    c2w: torch.Tensor         # (B,4,4)
    intrinsics: torch.Tensor  # (B,3,3)
    times: torch.Tensor       # (B,)
    time_idx: torch.Tensor    # (B,)
    categories: List[str]

    def to(self, device: str | torch.device) -> "SceneBatch":
        self.images = self.images.to(device)
        self.c2w = self.c2w.to(device)
        self.intrinsics = self.intrinsics.to(device)
        self.times = self.times.to(device)
        self.time_idx = self.time_idx.to(device)
        return self


def collate_scenes(samples: List[SceneSample]) -> SceneBatch:
    images = torch.stack([s.image for s in samples], 0)
    c2w = torch.stack([s.c2w for s in samples], 0)
    K = torch.stack([s.intrinsics for s in samples], 0)
    times = torch.tensor([s.time for s in samples], dtype=torch.float32)
    tidx = torch.tensor([s.time_idx for s in samples], dtype=torch.long)
    return SceneBatch(
        images=images, c2w=c2w, intrinsics=K,
        times=times, time_idx=tidx,
        categories=[s.category for s in samples],
    )
