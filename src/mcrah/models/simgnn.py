"""SIMGNN: Simplicial-style dense local-field propagation GNN (Stage 1).

Per workflow.md Phase 2 Step 6 / rules.md Rule 8, Stage 1 predicts dense local
physical offsets (Δposition, Δrotation) by propagating node features over the
hypergraph. It is the autoregressive dense-field propagator.

Design:
  * Each Gaussian's per-node feature = concat(position, scale, color SH,
    cluster-id embedding, temporal embedding). Built by :class:`FeatureEncoder`.
  * Hypergraph propagation (Θ) alternates with MLPs (HGNN block).
  * Residual updates with Gaussian-noise injection (rules.md Rule 4) on the
    *input* state to combat autoregressive drift.
  * Output heads predict delta-position and delta-rotation (as a quaternion
    delta, normalized).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureEncoder(nn.Module):
    """Encodes raw Gaussian attributes into the model feature space."""

    def __init__(self, feat_dim: int = 64, n_clusters: int = 256,
                 n_time_bins: int = 64):
        super().__init__()
        self.feat_dim = feat_dim
        self.proj = nn.Linear(3 + 3 + 3 + 1, feat_dim)  # pos(3)+scale(3)+sh(3)+op(1)
        self.cluster_emb = nn.Embedding(n_clusters + 1, feat_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, feat_dim), nn.SiLU(), nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, cloud, cluster_id: torch.Tensor, time: torch.Tensor
                ) -> torch.Tensor:
        # time: (N,1) or scalar -> (N,1)
        if time.dim() == 0:
            time = time.expand(cloud.means.shape[0], 1)
        elif time.dim() == 1:
            time = time.view(-1, 1)
        op = cloud.opacities[:, :1]
        raw = torch.cat([cloud.means, cloud.scales, cloud.sh, op], dim=-1)
        h = self.proj(raw)
        cid = cluster_id.clamp_min(0)
        h = h + self.cluster_emb(cid)
        h = h + self.time_mlp(time.to(h.dtype))
        return h


class HGNNBlock(nn.Module):
    """One hypergraph convolution block: Θ propagation + MLP + residual."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, hypergraph) -> torch.Tensor:
        # Normalized propagation (the Θ operator from hypergraph.py).
        propagated = hypergraph.propagate(h)
        h = h + self.drop(propagated)                 # residual
        h = h + self.drop(self._mlp(self.norm1(h)))
        return h

    def _propagate_mcrah(self, h: torch.Tensor, adaptive_hg) -> torch.Tensor:
        """Propagation through the MCRAH adaptive (soft) hypergraph.

        Same structure as ``forward`` but uses the soft-membership Θ operator
        so gradients flow through the membership gate.
        """
        propagated = adaptive_hg.propagate(h)
        h = h + self.drop(propagated)                 # residual
        h = h + self.drop(self._mlp(self.norm1(h)))
        return h

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class OffsetHeads(nn.Module):
    """Predicts Δposition (N,3) and Δrotation (N,4 quaternion delta).

    Rotations are predicted as small perturbations and normalized; the
    identity delta is (1,0,0,0)."""

    def __init__(self, dim: int, predict_rotation: bool = True,
                 pos_scale: float = 1e-2):
        super().__init__()
        self.predict_rotation = predict_rotation
        self.pos_scale = pos_scale
        self.pos_head = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 3)
        )
        if predict_rotation:
            # Predict a 3D axis-angle-like perturbation, convert to quaternion.
            self.rot_head = nn.Sequential(
                nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 3)
            )

    def forward(self, h: torch.Tensor):
        delta_pos = self.pos_scale * torch.tanh(self.pos_head(h))
        if self.predict_rotation:
            rotvec = self.rot_head(h)
            delta_rot = axis_angle_to_quaternion(rotvec)
        else:
            delta_rot = None
        return delta_pos, delta_rot


def axis_angle_to_quaternion(rotvec: torch.Tensor) -> torch.Tensor:
    """Convert (N,3) axis-angle to (N,4) quaternion (w,x,y,z), normalized.
    A near-zero rotvec yields the identity quaternion (1,0,0,0)."""
    angle = rotvec.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    axis = rotvec / angle
    half = angle * 0.5
    sin = torch.sin(half)
    cos = torch.cos(half)
    q = torch.cat([cos, axis * sin], dim=-1)
    # Identity when angle ~ 0 (numerically stable).
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=rotvec.device,
                            dtype=rotvec.dtype)
    near_zero = (angle < 1e-6).expand_as(q)
    return torch.where(near_zero, identity.expand_as(q), q)


class SIMGNN(nn.Module):
    """Stage-1 dense local-field propagation network.

    Takes the t=0 (or current autoregressive) Gaussian state + hypergraph and
    predicts per-node physical offsets for the next time step.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.model.feat_dim
        self.encoder = FeatureEncoder(
            feat_dim=d, n_clusters=max(256, 2 * int(d)),
        )
        self.blocks = nn.ModuleList([
            HGNNBlock(d, dropout=cfg.model.dropout)
            for _ in range(cfg.model.num_simgnn_layers)
        ])
        self.final_norm = nn.LayerNorm(d)
        self.heads = OffsetHeads(
            d, predict_rotation=cfg.model.predict_rotation
        )

    def encode(self, cloud, hypergraph, cluster_id: torch.Tensor,
               time: torch.Tensor) -> torch.Tensor:
        """Run the feature encoder + hypergraph propagation blocks.

        Stops *before* the offset heads so Stage-2 (DHGC) can refine the node
        features (workflow.md Phase 3 Step 7): ``h = SIMGNN(cloud, hg)`` then
        ``h = DHGC(h, dilated)`` then ``Δpos, Δrot = heads(h)``.
        """
        h = self.encoder(cloud, cluster_id, time)
        for blk in self.blocks:
            h = blk(h, hypergraph)
        return self.final_norm(h)

    def forward(self, cloud, hypergraph, cluster_id: torch.Tensor,
               time: torch.Tensor):
        """Encode features and predict offsets (standalone Stage-1 path).

        Returns ``(delta_pos, delta_rot)``. Callers that need to insert DHGC
        between propagation and the heads should use :meth:`encode` +
        :attr:`heads` instead (as :class:`MCRAH.step` does).
        """
        h = self.encode(cloud, hypergraph, cluster_id, time)
        return self.heads(h)
