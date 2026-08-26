"""DHGC: Dilated Hypergraph Construction (Stage 2 far-field attention).

Per workflow.md Phase 2 Step 5 / rules.md Rule 3. Stage 1 (SIMGNN) only sees
nodes within the same hyperedge, starving the receptive field. DHGC builds
multi-scale dilated neighbor sets (level k = nodes reachable in k edge-hops)
and applies cheap restricted self-attention over them, giving far-field
context at O(N * max_deg * D) instead of O(N^2).

This is the "Dynamic Far-Field Attention" module referenced in agent.md.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedAttention(nn.Module):
    """Restricted multi-head self-attention over a dilated neighbor set.

    For each node i, attends over its (padded) dilation-k neighbors using
    standard scaled-dot-product attention, masked to the valid neighbors.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, "feat_dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, idx: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (N, D) node features.
            idx: (N, max_deg) neighbor indices, -1 = padding.
            mask: (N, max_deg) bool, True = valid neighbor.
        Returns: (N, D) updated features (residual added).
        """
        N, D = x.shape
        H, Dh = self.num_heads, self.head_dim
        max_deg = idx.shape[1]

        q = self.q(x).view(N, H, Dh)                       # (N,H,Dh)

        # Gather neighbor keys/values. Replace -1 pad with 0 to avoid OOB.
        safe_idx = idx.clamp_min(0)
        k_all = self.k(x)[safe_idx]                        # (N,max_deg,D)
        v_all = self.v(x)[safe_idx]

        k = k_all.view(N, max_deg, H, Dh)
        v = v_all.view(N, max_deg, H, Dh)

        # (N,H,1,Dh) x (N,H,Dh,max_deg) -> (N,H,1,max_deg)
        attn = torch.einsum("nhd,nmhd->nhm", q, k) * self.scale
        # Mask padding neighbors with -inf.
        neg = torch.finfo(attn.dtype).min
        attn = attn.masked_fill(~mask.unsqueeze(1), neg)
        attn = attn.softmax(dim=-1)                         # (N,H,1,max_deg)
        attn = self.drop(attn)

        # Weighted sum: (N,H,1,max_deg) x (N,H,max_deg,Dh) -> (N,H,Dh)
        out = torch.einsum("nhm,nmhd->nhd", attn, v)
        out = out.reshape(N, D)
        return self.out(out)


class DHGCBlock(nn.Module):
    """One DHGC layer: dilated attention + FFN + residual, for one level."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = DilatedAttention(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, idx: torch.Tensor, mask: torch.Tensor
                ) -> torch.Tensor:
        a = self.attn(self.norm1(x), idx, mask)
        x = x + self.drop(a)
        x = x + self.drop(self.fc2(F.gelu(self.fc1(self.norm2(x)))))
        return x


class DHGC(nn.Module):
    """Dilated Hypergraph Construction module (Stage 2 far-field attention).

    Applies :class:`DHGCBlock` at each dilation level, optionally fusing the
    multi-scale outputs. We process levels in ascending order so coarser
    (far-field) context refines the fine propagation.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.model.feat_dim
        self.levels = cfg.hypergraph.dilation_levels
        # One block per level; weights are distinct so the network can learn
        # scale-specific far-field patterns.
        self.blocks = nn.ModuleDict({
            str(k): DHGCBlock(d, cfg.model.num_heads, cfg.model.dropout)
            for k in self.levels
        })
        self.fuse = nn.Linear(d * len(self.levels), d) if len(self.levels) > 1 \
            else nn.Identity()
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, dilated) -> torch.Tensor:
        """Args:
            x: (N, D) node features (post-SIMGNN).
            dilated: :class:`DilatedStructure` with padded neighbor sets.
        Returns: (N, D) refined features.
        """
        outs = []
        for k in self.levels:
            idx = dilated.padded[k].to(x.device)
            mask = dilated.mask[k].to(x.device)
            outs.append(self.blocks[str(k)](x, idx, mask))
        if len(self.levels) > 1:
            out = self.fuse(torch.cat(outs, dim=-1))
        else:
            out = outs[0]
        return self.norm(out)
