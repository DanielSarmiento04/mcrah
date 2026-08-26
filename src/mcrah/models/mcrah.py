"""MCRAHCore: Motion-Coherent Rigidity-Adaptive Hypergraph.

The novel contribution beyond DVHGNN (CVPR 2025). The static cosine-similarity
hypergraph of DVHGNN assumes cluster membership is fixed at t=0 — but in a
dynamic 3DGS scene, rigid body parts split and merge as motion diverges.
A trex's tail is one coherent cluster at rest, yet the base and tip move
differently as it swings.

MCRAHCore adds three things that no published work combines:

1. **Motion-coherent topology evolution** — a learned gate computes a soft
   membership matrix M (N, E) at *each* autoregressive step from the current
   deformation state, not from static appearance similarity. The hypergraph
   structure evolves with the scene.

2. **Rigidity prior loss** — within each soft hyperedge, per-node displacement
   is pulled toward the cluster-centroid displacement (quasi-rigid body
   assumption). This is physics-grounded: a hyperedge represents a quasi-rigid
   part, and the loss penalizes intra-cluster deformation deviation.

3. **Topology temporal smoothness** — membership reassignment is regularized so
   the hyperedge structure does not jump abruptly between time steps, ensuring
   temporal stability of the learned topology.

The soft membership M replaces the hard incidence H in the Θ propagation
operator, making the topology *differentiable* and trainable end-to-end through
the autoregressive rollout.

Soft Θ operator (generalization of HGNN, Feng et al. 2019):

    Θ = D_v^{-1/2} M W D_e^{-1} M^T D_v^{-1/2}

where D_v = diag(M 1_E), D_e = diag(M^T 1_N), and M ∈ [0,1]^{N×E}.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Soft membership gate
# --------------------------------------------------------------------------- #
class MCRAHGate(nn.Module):
    """Computes a soft membership matrix M (N, E) from the current motion state.

    For each node i and cluster e, a compatibility score is computed from the
    node's displacement and the cluster's centroid displacement. A learnable
    prior biases toward the t=0 assignment (rigidity reference). The output
    is a softmax over clusters, yielding a differentiable soft membership.

    Args:
        feat_dim: internal feature dimension for the motion encoders.
        n_clusters: number of hyperedges E (= initial cluster count).
        tau: temperature for the membership softmax.
    """

    def __init__(self, feat_dim: int = 64, n_clusters: int = 256, tau: float = 1.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.n_clusters = n_clusters
        self.tau = tau

        # Node motion encoder: displacement (3) + reference position (3) -> D
        self.node_enc = nn.Sequential(
            nn.Linear(6, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, feat_dim),
        )
        # Cluster motion encoder: centroid displacement (3) + ref centroid (3) -> D
        self.cluster_enc = nn.Sequential(
            nn.Linear(6, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, feat_dim),
        )
        # Learnable prior strength: how much to trust the initial assignment.
        # Initialized high so the model starts close to the static hypergraph
        # and gradually learns to deviate where motion demands it.
        self.prior_logit = nn.Parameter(torch.tensor(3.0))

    def forward(
        self,
        means_t: torch.Tensor,           # (N, 3) current positions
        means_0: torch.Tensor,           # (N, 3) reference (t=0) positions
        assignment_0: torch.Tensor,      # (N,) long, initial cluster id
        centroid_0: torch.Tensor,        # (E, 3) reference centroids
    ) -> torch.Tensor:
        """Returns M (N, E) soft membership in [0, 1], rows sum to 1."""
        N = means_t.shape[0]
        E = self.n_clusters

        # Cluster centroids at current time (using initial assignment for scatter).
        centroid_t = _scatter_mean(means_t, assignment_0, E)  # (E, 3)

        # Node displacement + reference position.
        node_disp = means_t - means_0                     # (N, 3)
        node_feat = self.node_enc(
            torch.cat([node_disp, means_0], dim=-1))       # (N, D)

        # Cluster displacement + reference centroid.
        cluster_disp = centroid_t - centroid_0             # (E, 3)
        cluster_feat = self.cluster_enc(
            torch.cat([cluster_disp, centroid_0], dim=-1)) # (E, D)

        # Compatibility: dot product (N, E).
        logits = node_feat @ cluster_feat.t() / self.tau   # (N, E)

        # Prior bias: strong logit for the initial assignment, negative for others.
        # Initialized high so the model starts close to the static hypergraph
        # and gradually learns to deviate where motion demands it. Kept
        # differentiable so prior_logit can be learned end-to-end.
        mask_0 = F.one_hot(
            assignment_0.clamp_min(0).clamp_max(E - 1), num_classes=E
        ).to(logits.dtype)                                   # (N, E)
        prior = self.prior_logit * (2.0 * mask_0 - 1.0)     # +pl assigned, -pl else
        logits = logits + prior

        return logits.softmax(dim=-1)                       # (N, E)


# --------------------------------------------------------------------------- #
# Adaptive hypergraph (soft propagation)
# --------------------------------------------------------------------------- #
class AdaptiveHypergraph(nn.Module):
    """Differentiable hypergraph with soft membership M.

    Replaces the hard incidence matrix H of :class:`Hypergraph` with a soft
    membership M ∈ [0,1]^{N×E}. The Θ propagation operator becomes:

        Θ = D_v^{-1/2} M W D_e^{-1} M^T D_v^{-1/2}

    Implemented as three matmuls (N×E is small since E = O(√N)), so the soft
    propagation is efficient and fully differentiable through M.

    When ``adaptive=False``, M is the one-hot assignment and propagation
    reduces to the original hard-incidence HGNN operator (ablation arm).
    """

    def __init__(self, assignment_0: torch.Tensor, n_clusters: int, adaptive: bool = True):
        super().__init__()
        self.n_nodes = assignment_0.shape[0]
        self.n_edges = n_clusters
        self.adaptive = adaptive

        # Store the reference (t=0) assignment as a non-persistent buffer.
        self.register_buffer("assignment_0", assignment_0.clone(), persistent=False)
        # Current membership (updated at each step). Init to one-hot.
        one_hot = F.one_hot(
            assignment_0.clamp_min(0).clamp_max(n_clusters - 1),
            num_classes=n_clusters,
        ).float()
        self.register_buffer("membership", one_hot, persistent=False)

    def set_membership(self, M: torch.Tensor) -> None:
        """Update the soft membership matrix (called at each rollout step)."""
        self.membership = M

    def to(self, device) -> "AdaptiveHypergraph":
        self.assignment_0 = self.assignment_0.to(device)
        self.membership = self.membership.to(device)
        return self

    def propagate(self, x: torch.Tensor, edge_weight: Optional[torch.Tensor] = None
                  ) -> torch.Tensor:
        """Apply the soft Θ operator to node features x (N, D).

        Θ x = D_v^{-1/2} M W D_e^{-1} M^T D_v^{-1/2} x
        """
        M = self.membership                              # (N, E)
        E = self.n_edges
        dev = x.device

        if edge_weight is None:
            w = torch.ones(E, device=dev, dtype=x.dtype)
        else:
            w = edge_weight.to(dev).to(x.dtype)

        # Vertex / edge degrees under soft membership.
        d_v = M.sum(dim=1).clamp_min(1.0)                 # (N,)
        d_e = M.sum(dim=0).clamp_min(1.0)                 # (E,)

        # 1. D_v^{-1/2} x
        dv_inv_sqrt = (1.0 / d_v).sqrt()
        x1 = x * dv_inv_sqrt.unsqueeze(-1)               # (N, D)

        # 2. M^T x1  (node -> edge): (E, N) x (N, D) = (E, D)
        edge_feats = M.t() @ x1

        # 3. W D_e^{-1}
        de_inv = 1.0 / d_e
        edge_feats = edge_feats * (w * de_inv).unsqueeze(-1)  # (E, D)

        # 4. M (edge_feats)  (edge -> node): (N, E) x (E, D) = (N, D)
        node_feats = M @ edge_feats

        # 5. D_v^{-1/2}
        return node_feats * dv_inv_sqrt.unsqueeze(-1)


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
class RigidityLoss(nn.Module):
    """Quasi-rigid-body prior on soft hyperedge membership.

    For each node i, its displacement should match the average displacement of
    its (soft) cluster. Under rigid translation this is exact; the loss
    penalizes intra-cluster deformation deviation.

        L_rig = Σ_i || Δ_i - Σ_e M[i,e] · Δ̄_e ||²

    where Δ_i = means_t[i] - means_0[i] and Δ̄_e = centroid displacement of e.
    The loss is differentiable through M, so the gate learns to form clusters
    that move coherently.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        means_t: torch.Tensor,          # (N, 3) current positions
        means_0: torch.Tensor,          # (N, 3) reference positions
        membership: torch.Tensor,        # (N, E) soft membership
        assignment_0: torch.Tensor,      # (N,) initial assignment (for centroid)
    ) -> torch.Tensor:
        E = membership.shape[1]
        node_disp = means_t - means_0                    # (N, 3)

        # Per-cluster centroid displacement (using initial assignment for scatter,
        # so the centroids are stable regardless of current soft membership).
        centroid_disp = _scatter_mean(node_disp, assignment_0, E)  # (E, 3)

        # Expected displacement under soft membership: M @ centroid_disp.
        expected = membership @ centroid_disp             # (N, 3)

        residual = node_disp - expected                  # (N, 3)
        return (residual ** 2).sum(dim=-1).mean()


class TopologySmoothnessLoss(nn.Module):
    """Penalizes abrupt changes in hyperedge membership between time steps.

        L_topo = || M_t - M_{t-1} ||_F² / N

    Encourages the topology to evolve smoothly, preventing the gate from
    rapidly oscillating cluster assignments.
    """

    def __init__(self):
        super().__init__()

    def forward(self, M_t: torch.Tensor, M_prev: torch.Tensor) -> torch.Tensor:
        return ((M_t - M_prev) ** 2).sum() / M_t.shape[0]


# --------------------------------------------------------------------------- #
# Full MCRAH core module
# --------------------------------------------------------------------------- #
class MCRAHCore(nn.Module):
    """Motion-Coherent Rigidity-Adaptive Hypergraph core.

    Wraps the gate + adaptive hypergraph. At each autoregressive step, the
    gate recomputes soft membership from the current deformation state, and
    the adaptive hypergraph propagates features through it.

    The reference (t=0) centroids and assignment are fixed from Phase 1/2.
    """

    def __init__(
        self,
        assignment_0: torch.Tensor,
        n_clusters: int,
        feat_dim: int = 64,
        tau: float = 1.0,
        adaptive: bool = True,
    ):
        super().__init__()
        self.adaptive = adaptive
        self.gate = MCRAHGate(feat_dim=feat_dim, n_clusters=n_clusters, tau=tau)
        self.hypergraph = AdaptiveHypergraph(
            assignment_0, n_clusters, adaptive=adaptive)
        # Store reference centroids (computed externally from t=0 means).
        self.register_buffer("centroid_0", torch.zeros(n_clusters, 3), persistent=False)
        self.register_buffer("means_0", torch.zeros(assignment_0.shape[0], 3),
                             persistent=False)

    def set_reference(self, means_0: torch.Tensor, centroid_0: torch.Tensor) -> None:
        """Store the t=0 reference state (called once at init)."""
        self.means_0 = means_0.detach().clone()
        self.centroid_0 = centroid_0.detach().clone()

    def update_membership(
        self,
        means_t: torch.Tensor,
    ) -> torch.Tensor:
        """Recompute soft membership from the current positions.

        Returns M (N, E). When ``adaptive=False``, returns the one-hot
        initial assignment (no gate computation — ablation arm).
        """
        if not self.adaptive:
            M = F.one_hot(
                self.hypergraph.assignment_0.clamp_min(0),
                num_classes=self.hypergraph.n_edges,
            ).float()
            self.hypergraph.set_membership(M)
            return M

        M = self.gate(means_t, self.means_0,
                       self.hypergraph.assignment_0, self.centroid_0)
        self.hypergraph.set_membership(M)
        return M

    def propagate(self, x: torch.Tensor, edge_weight: Optional[torch.Tensor] = None
                  ) -> torch.Tensor:
        """Delegate to the adaptive hypergraph's soft propagation."""
        return self.hypergraph.propagate(x, edge_weight)

    def to(self, device) -> "MCRAHCore":
        self.gate = self.gate.to(device)
        self.hypergraph = self.hypergraph.to(device)
        self.centroid_0 = self.centroid_0.to(device)
        self.means_0 = self.means_0.to(device)
        return self


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int
                  ) -> torch.Tensor:
    """Mean of ``src`` (N, D) grouped by ``index`` (N,) into ``dim_size`` groups.

    Returns (dim_size, D).
    """
    out = torch.zeros(dim_size, src.shape[-1], device=src.device, dtype=src.dtype)
    count = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    count.index_add_(0, index, torch.ones(src.shape[0], 1,
                                           device=src.device, dtype=src.dtype))
    return out / count.clamp_min(1.0)


def compute_centroids(means: torch.Tensor, assignment: torch.Tensor,
                      n_clusters: int) -> torch.Tensor:
    """Compute cluster centroids (E, 3) from means (N, 3) + assignment (N,)."""
    return _scatter_mean(means, assignment.clamp_min(0), n_clusters)
