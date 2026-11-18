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
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..gs.gaussian import rotation_to_matrix
from .simgnn import axis_angle_to_quaternion


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
        # Temporal conditioning so soft membership adapts to target time t
        # even at step 0 when means_t == means_0.
        self.time_enc = nn.Sequential(
            nn.Linear(5, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, feat_dim),
        )
        nn.init.zeros_(self.time_enc[-1].weight)
        nn.init.zeros_(self.time_enc[-1].bias)
        self.register_buffer("prior_logit", torch.tensor(3.0))

    def forward(
        self,
        means_t: torch.Tensor,           # (N, 3) current positions
        means_0: torch.Tensor,           # (N, 3) reference (t=0) positions
        assignment_0: torch.Tensor,      # (N,) long, initial cluster id
        centroid_0: torch.Tensor,        # (E, 3) reference centroids
        time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns M (N, E) soft membership in [0, 1], rows sum to 1."""
        N = means_t.shape[0]
        E = self.n_clusters

        # Cluster centroids at current time (using initial assignment for scatter).
        centroid_t = _scatter_mean(means_t, assignment_0, E)  # (E, 3)

        # Node displacement + reference position (bounded for rollout stability).
        node_disp = torch.tanh((means_t - means_0) * 5.0) * 0.2  # (N, 3)
        node_feat = self.node_enc(
            torch.cat([node_disp, means_0], dim=-1))       # (N, D)

        if time is not None:
            t_val = time.to(device=means_t.device, dtype=means_t.dtype).reshape(-1)
            t_s = t_val[:1] if t_val.numel() > 0 else torch.zeros(1, device=means_t.device, dtype=means_t.dtype)
            t_emb = torch.cat([
                t_s,
                torch.sin(t_s * math.pi),
                torch.cos(t_s * math.pi),
                torch.sin(t_s * 2.0 * math.pi),
                torch.cos(t_s * 2.0 * math.pi),
            ], dim=-1).unsqueeze(0)  # (1, 5)
            node_feat = node_feat * (1.0 + 0.25 * torch.tanh(self.time_enc(t_emb)))

        # Cluster displacement + reference centroid.
        cluster_disp = torch.tanh((centroid_t - centroid_0) * 5.0) * 0.2  # (E, 3)
        cluster_feat = self.cluster_enc(
            torch.cat([cluster_disp, centroid_0], dim=-1)) # (E, D)

        # Compatibility: scaled dot product (N, E) performed in float32 for AMP stability.
        scale = max(1e-4, float(self.tau) * (self.feat_dim ** 0.5))
        logits = (node_feat.float() @ cluster_feat.float().t()) / scale

        # Prior bias: strong logit for the initial assignment, negative for others.
        mask_0 = F.one_hot(
            assignment_0.clamp_min(0).clamp_max(E - 1), num_classes=E
        ).float()                                            # (N, E)
        pl = self.prior_logit.clamp(-10.0, 10.0).float()
        prior = pl * (2.0 * mask_0 - 1.0)     # +pl assigned, -pl else
        logits = logits + prior

        res = logits.softmax(dim=-1).to(node_feat.dtype)    # (N, E)
        return torch.nan_to_num(res, nan=1.0 / E)


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
        orig_dtype = x.dtype
        x_f = x.float()
        M = self.membership.float()                      # (N, E)
        E = self.n_edges
        dev = x.device

        if edge_weight is None:
            w = torch.ones(E, device=dev, dtype=torch.float32)
        else:
            w = edge_weight.to(dev).float()

        # Vertex / edge degrees under soft membership.
        d_v = M.sum(dim=1).clamp_min(1.0)                 # (N,)
        d_e = M.sum(dim=0).clamp_min(1.0)                 # (E,)

        # 1. D_v^{-1/2} x
        dv_inv_sqrt = (1.0 / d_v).sqrt()
        x1 = x_f * dv_inv_sqrt.unsqueeze(-1)             # (N, D)

        # 2. M^T x1  (node -> edge): (E, N) x (N, D) = (E, D)
        edge_feats = M.t() @ x1

        # 3. W D_e^{-1}
        de_inv = 1.0 / d_e
        edge_feats = edge_feats * (w * de_inv).unsqueeze(-1)  # (E, D)

        # 4. M (edge_feats)  (edge -> node): (N, E) x (E, D) = (N, D)
        node_feats = M @ edge_feats

        # 5. D_v^{-1/2}
        out = node_feats * dv_inv_sqrt.unsqueeze(-1)
        return out.to(orig_dtype)


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
        expected_disp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        E = membership.shape[1]
        node_disp = means_t.float() - means_0.float()    # (N, 3)
        if expected_disp is not None:
            # SO(3)-invariant SE(3) rigidity: penalize only non-rigid residual
            # beyond the hyperedge SE(3) rigid-body articulation field.
            residual = node_disp - expected_disp.float()
            res = (residual ** 2).sum(dim=-1).mean()
            return torch.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)

        M_f = membership.float()

        # Per-cluster centroid displacement (using initial assignment for scatter,
        # so the centroids are stable regardless of current soft membership).
        centroid_disp = _scatter_mean(node_disp, assignment_0, E)  # (E, 3)

        # Expected displacement under soft membership: M @ centroid_disp.
        expected = M_f @ centroid_disp                   # (N, 3)

        residual = node_disp - expected                  # (N, 3)
        res = (residual ** 2).sum(dim=-1).mean()
        return torch.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)


class OrthogonalityLoss(nn.Module):
    """Hyperedge subspace orthogonality regularizer: || M_norm^T M_norm - I_E ||_F^2 / E.

    Encourages the soft membership columns to span decoupled, orthogonal
    kinematic subspaces so independent rigid bodies (e.g. bouncing spheres vs.
    floor plate, or left vs. right limbs) do not mix during propagation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, membership: torch.Tensor) -> torch.Tensor:
        M = membership.float()                           # (N, E)
        col_norms = M.norm(dim=0, keepdim=True).clamp_min(1e-6)
        M_norm = M / col_norms
        gram = M_norm.t() @ M_norm                       # (E, E)
        E = M.shape[1]
        eye = torch.eye(E, device=M.device, dtype=torch.float32)
        res = ((gram - eye) ** 2).sum() / max(1, E)
        return torch.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)


class HypergraphJEPAPredictor(nn.Module):
    """Hypergraph Joint-Embedding Predictive Architecture (Hypergraph-JEPA).

    Pools node features into hyperedge latents z_e(t) in R^{E x D}, predicts
    future hyperedge latents z_hat_e(t + dt) from (z_e(t), dt) in abstract
    embedding space (with stop-gradient on the target), and applies VICReg
    variance + off-diagonal covariance orthogonality to prevent representation
    collapse.
    """

    def __init__(self, feat_dim: int = 64):
        super().__init__()
        self.feat_dim = feat_dim
        self.pred_mlp = nn.Sequential(
            nn.LayerNorm(feat_dim + 1),
            nn.Linear(feat_dim + 1, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, feat_dim),
        )
        nn.init.normal_(self.pred_mlp[-1].weight, std=1e-3)
        nn.init.zeros_(self.pred_mlp[-1].bias)

    @staticmethod
    def pool_latents(h: torch.Tensor, membership: torch.Tensor) -> torch.Tensor:
        M = membership.float()
        d_e = M.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
        return (M.t() @ h.float()) / d_e                 # (E, D)

    def forward(self, z_t: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        E = z_t.shape[0]
        dt_col = dt.to(device=z_t.device, dtype=z_t.dtype).reshape(1, 1).expand(E, 1)
        return z_t + self.pred_mlp(torch.cat([z_t, dt_col], dim=-1))

    def jepa_vicreg_loss(
        self,
        z_curr: torch.Tensor,
        z_next: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        z_pred = self.forward(z_curr, dt)
        z_tgt = z_next.detach()
        pred_err = F.mse_loss(z_pred, z_tgt)

        # VICReg covariance orthogonality across latent feature channels (D x D)
        E, D = z_curr.shape
        if E > 1:
            zc = z_curr - z_curr.mean(dim=0, keepdim=True)
            cov = (zc.t() @ zc) / max(1, E - 1)
            off_diag = cov - torch.diag(torch.diag(cov))
            cov_loss = (off_diag ** 2).sum() / max(1, D)
            std = torch.sqrt(z_curr.var(dim=0, unbiased=False) + 1e-4)
            var_loss = F.relu(0.5 - std).mean()
        else:
            cov_loss = z_curr.new_zeros(())
            var_loss = z_curr.new_zeros(())
        res = pred_err + 0.05 * cov_loss + 0.05 * var_loss
        return torch.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)


class TopologySmoothnessLoss(nn.Module):
    """Penalizes abrupt changes in hyperedge membership between time steps.

        L_topo = || M_t - M_{t-1} ||_F² / N

    Encourages the topology to evolve smoothly, preventing the gate from
    rapidly oscillating cluster assignments.
    """

    def __init__(self):
        super().__init__()

    def forward(self, M_t: torch.Tensor, M_prev: torch.Tensor) -> torch.Tensor:
        diff = M_t.float() - M_prev.float()
        res = (diff ** 2).sum() / max(1, M_t.shape[0])
        return torch.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------- #
# Full MCRAH core module
# --------------------------------------------------------------------------- #
class HyperedgeSE3Articulator(nn.Module):
    """Per-hyperedge SE(3) rigid-body articulation (Soft Hypergraph LBS).

    Pools refined node features h (N, D) into hyperedge features h_e (E, D)
    via the soft membership M (N, E), predicts a per-hyperedge 3D rotation
    R_e in SO(3) around the canonical cluster pivot c_e^0 and a 3D translation
    b_e, and blends the articulated rigid displacement across nodes in O(N*E):

        Delta_x_i^{SE3} = sum_e M_{ie} [ (R_e - I)(x_i^0 - c_e^0) + b_e ]
    """

    def __init__(self, feat_dim: int = 64, max_trans: float = 0.35, max_rot: float = 0.85):
        super().__init__()
        self.max_trans = max_trans
        self.max_rot = max_rot
        self.edge_mlp = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, 6),  # 3 translation + 3 axis-angle rotation
        )
        nn.init.normal_(self.edge_mlp[-1].weight, std=1e-3)
        nn.init.zeros_(self.edge_mlp[-1].bias)

    def forward(
        self,
        h: torch.Tensor,            # (N, D) post-DHGC node features
        membership: torch.Tensor,   # (N, E) soft hyperedge membership
        means_0: torch.Tensor,      # (N, 3) canonical t=0 node positions
        centroid_0: torch.Tensor,   # (E, 3) canonical t=0 hyperedge pivots
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (delta_pos_se3 (N,3), rotvec_se3 (N,3))."""
        orig_dtype = h.dtype
        h_f = h.float()
        M = membership.float()      # (N, E)
        x0 = means_0.float()        # (N, 3)
        c0 = centroid_0.float()     # (E, 3)

        # 1. Pool node features into hyperedges: (E, N) @ (N, D) -> (E, D)
        d_e = M.sum(dim=0).clamp_min(1.0).unsqueeze(-1)  # (E, 1)
        h_edge = (M.t() @ h_f) / d_e                     # (E, D)

        # 2. Predict per-hyperedge SE(3) parameters (b_e, omega_e)
        raw_se3 = self.edge_mlp(h_edge)                  # (E, 6)
        b_e = self.max_trans * torch.tanh(raw_se3[:, :3])      # (E, 3)
        omega_e = self.max_rot * torch.tanh(raw_se3[:, 3:])    # (E, 3)

        # 3. Convert axis-angle omega_e -> rotation matrix R_e (E, 3, 3)
        q_e = axis_angle_to_quaternion(omega_e)          # (E, 4)
        R_e = rotation_to_matrix(q_e)                    # (E, 3, 3)
        eye = torch.eye(3, device=h.device, dtype=torch.float32).unsqueeze(0)
        A_e = R_e - eye                                  # (E, 3, 3)

        # 4. Factorized O(N*E) Linear Blend Skinning:
        # sum_e M_{ie} [ A_e (x_i^0 - c_e^0) + b_e ] = (sum_e M_{ie} A_e) x_i^0 + sum_e M_{ie} (b_e - A_e c_e^0)
        Ac0 = torch.bmm(A_e, c0.unsqueeze(-1)).squeeze(-1)     # (E, 3)
        b_tilde_e = b_e - Ac0                                  # (E, 3)

        A_node = (M @ A_e.reshape(-1, 9)).reshape(-1, 3, 3)    # (N, 3, 3)
        b_node = M @ b_tilde_e                                 # (N, 3)

        delta_pos_se3 = torch.bmm(A_node, x0.unsqueeze(-1)).squeeze(-1) + b_node
        rotvec_se3 = M @ omega_e                               # (N, 3)

        delta_pos_se3 = torch.nan_to_num(delta_pos_se3, nan=0.0).to(orig_dtype)
        rotvec_se3 = torch.nan_to_num(rotvec_se3, nan=0.0).to(orig_dtype)
        return delta_pos_se3, rotvec_se3


class MCRAHCore(nn.Module):
    """Motion-Coherent Rigidity-Adaptive Hypergraph core.

    Wraps the gate + adaptive hypergraph + SE(3) hyperedge articulator.
    At each autoregressive step, the gate recomputes soft membership from the
    current deformation state, the adaptive hypergraph propagates features
    through it, and the SE(3) articulator computes rigid-body rotations and
    translations around each hyperedge's canonical pivot.

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
        self.articulator = HyperedgeSE3Articulator(feat_dim=feat_dim)
        self.jepa = HypergraphJEPAPredictor(feat_dim=feat_dim)
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
        time: Optional[torch.Tensor] = None,
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
                       self.hypergraph.assignment_0, self.centroid_0,
                       time=time)
        self.hypergraph.set_membership(M)
        return M

    def articulate(self, h: torch.Tensor, membership: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-node SE(3) articulated displacement & rotation vector."""
        return self.articulator(h, membership, self.means_0, self.centroid_0)

    def propagate(self, x: torch.Tensor, edge_weight: Optional[torch.Tensor] = None
                  ) -> torch.Tensor:
        """Delegate to the adaptive hypergraph's soft propagation."""
        return self.hypergraph.propagate(x, edge_weight)

    def to(self, device) -> "MCRAHCore":
        self.gate = self.gate.to(device)
        self.hypergraph = self.hypergraph.to(device)
        self.articulator = self.articulator.to(device)
        self.jepa = self.jepa.to(device)
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
    idx = index.clamp_min(0).clamp_max(max(0, dim_size - 1))
    out = torch.zeros(dim_size, src.shape[-1], device=src.device, dtype=src.dtype)
    count = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    out.index_add_(0, idx, src)
    count.index_add_(0, idx, torch.ones(src.shape[0], 1,
                                         device=src.device, dtype=src.dtype))
    res = out / count.clamp_min(1.0)
    return torch.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)


def compute_centroids(means: torch.Tensor, assignment: torch.Tensor,
                      n_clusters: int) -> torch.Tensor:
    """Compute cluster centroids (E, 3) from means (N, 3) + assignment (N,)."""
    return _scatter_mean(means, assignment.clamp_min(0), n_clusters)
