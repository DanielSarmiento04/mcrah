"""Hypergraph data structures and propagation operators.

This module is the Python counterpart to the Rust clustering preprocessor
(``rust/src/clustering.rs`` + ``rust/src/graph.rs``). It provides:

  * :class:`Hypergraph`        - incidence structure + normalized propagation.
  * :class:`DilatedStructure`  - multi-scale CSR dilated neighbor sets (DHGC).
  * loaders for the Rust-preprocessed ``.npy`` tensors, plus a pure-Python
    fallback so the pipeline runs before preprocessing has been executed.

Propagation math (HGNN, Feng et al. 2019):
    Θ = D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}
    X' = σ(Θ X Θ_weight)

Θ is applied as two sparse scatter steps (node→edge→node) so we never
materialize the N×N matrix — essential for 50k+ Gaussians (rules.md Rule 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Core hypergraph
# --------------------------------------------------------------------------- #
@dataclass
class Hypergraph:
    """A static hypergraph G=(V,E) over N nodes and E hyperedges.

    Stored sparsely as edge membership lists; the normalized propagation
    operator Θ is applied on-the-fly via scatter.
    """
    n_nodes: int
    n_edges: int
    assignment: torch.Tensor           # (N,) long, node -> edge id
    edge_lists: List[List[int]]         # E lists of member node ids

    # Precomputed flat (node, edge) incidence pairs for fast scatter.
    _node_idx: torch.Tensor = field(default=None, repr=False)   # (nnz,)
    _edge_idx: torch.Tensor = field(default=None, repr=False)   # (nnz,)
    _d_v: torch.Tensor = field(default=None, repr=False)        # (N,) vertex degree
    _d_e: torch.Tensor = field(default=None, repr=False)        # (E,) edge degree

    def __post_init__(self):
        self._build_indices()

    def _build_indices(self):
        """Flatten edge_lists into parallel (node, edge) index arrays and
        compute degree vectors."""
        nodes, edges = [], []
        for e, members in enumerate(self.edge_lists):
            for m in members:
                nodes.append(m)
                edges.append(e)
        device = self.assignment.device
        self._node_idx = torch.tensor(nodes, dtype=torch.long, device=device)
        self._edge_idx = torch.tensor(edges, dtype=torch.long, device=device)
        # Degrees via bincount.
        self._d_v = torch.bincount(self._node_idx, minlength=self.n_nodes).float()
        self._d_e = torch.bincount(self._edge_idx, minlength=self.n_edges).float()

    def to(self, device) -> "Hypergraph":
        return Hypergraph(
            n_nodes=self.n_nodes, n_edges=self.n_edges,
            assignment=self.assignment.to(device),
            edge_lists=self.edge_lists,
            _node_idx=self._node_idx.to(device),
            _edge_idx=self._edge_idx.to(device),
            _d_v=self._d_v.to(device),
            _d_e=self._d_e.to(device),
        )

    def propagate(self, x: torch.Tensor, edge_weight: Optional[torch.Tensor] = None
                  ) -> torch.Tensor:
        """Apply the normalized hypergraph operator Θ to node features x.

        Θ x = D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2} x

        Args:
            x: (N, D) node features.
            edge_weight: (E,) optional per-edge weights W (default ones).
        Returns: (N, D) propagated features.
        """
        N, E = self.n_nodes, self.n_edges
        dev = x.device
        d_v = self._d_v.to(dev).clamp_min(1.0)
        d_e = self._d_e.to(dev).clamp_min(1.0)
        node_idx = self._node_idx.to(dev)
        edge_idx = self._edge_idx.to(dev)

        if edge_weight is None:
            w = torch.ones(E, device=dev, dtype=x.dtype)
        else:
            w = edge_weight.to(dev).to(x.dtype)

        # 1. D_v^{-1/2} x
        dv_inv_sqrt = (1.0 / d_v).sqrt()
        x1 = x * dv_inv_sqrt.unsqueeze(-1)

        # 2. H^T x1  (node -> edge): aggregate node features into edges.
        edge_feats = torch.zeros(E, x.shape[-1], device=dev, dtype=x.dtype)
        edge_feats.index_add_(0, edge_idx, x1[node_idx])

        # 3. W D_e^{-1} (edge_feats)
        de_inv = 1.0 / d_e
        edge_feats = edge_feats * (w * de_inv).unsqueeze(-1)

        # 4. H (edge_feats)  (edge -> node): scatter back to nodes.
        node_feats = torch.zeros(N, x.shape[-1], device=dev, dtype=x.dtype)
        node_feats.index_add_(0, node_idx, edge_feats[edge_idx])

        # 5. D_v^{-1/2}
        return node_feats * dv_inv_sqrt.unsqueeze(-1)

    def incidence_dense(self) -> torch.Tensor:
        """Dense (N, E) incidence matrix. For small graphs / debugging only."""
        H = torch.zeros(self.n_nodes, self.n_edges)
        for e, members in enumerate(self.edge_lists):
            for m in members:
                H[m, e] = 1.0
        return H


# --------------------------------------------------------------------------- #
# Dilated structure (DHGC)
# --------------------------------------------------------------------------- #
@dataclass
class DilatedStructure:
    """Multi-scale dilated neighbor sets for far-field attention (DHGC).

    For each dilation level k, stores a padded (N, max_deg) neighbor index
    built from the Rust-exported CSR arrays, plus a validity mask. Attention
    is restricted to these neighbors, giving O(N * max_deg * D) cost instead
    of O(N^2) (rules.md Rule 3).
    """
    levels: Tuple[int, ...]
    # padded_idx[k] : (N, max_deg_k) long, -1 = padding
    padded: Dict[int, torch.Tensor]
    # mask[k] : (N, max_deg_k) bool
    mask: Dict[int, torch.Tensor]

    def to(self, device) -> "DilatedStructure":
        return DilatedStructure(
            levels=self.levels,
            padded={k: v.to(device) for k, v in self.padded.items()},
            mask={k: v.to(device) for k, v in self.mask.items()},
        )


def _csr_to_padded(neighbors: np.ndarray, offsets: np.ndarray, n_nodes: int,
                   max_neighbors: int, seed: int = 42
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert CSR (neighbors, offsets) to a padded (N, max_deg) index + mask.

    Nodes whose degree exceeds ``max_neighbors`` are subsampled (deterministic,
    seeded) to bound the attention cost.
    """
    rng = np.random.default_rng(seed)
    idx = np.full((n_nodes, max_neighbors), -1, dtype=np.int64)
    mask = np.zeros((n_nodes, max_neighbors), dtype=bool)
    for i in range(n_nodes):
        s, e = int(offsets[i]), int(offsets[i + 1])
        nb = neighbors[s:e]
        if len(nb) > max_neighbors:
            choice = rng.choice(len(nb), size=max_neighbors, replace=False)
            nb = nb[choice]
        idx[i, :len(nb)] = nb
        mask[i, :len(nb)] = True
    return (torch.from_numpy(idx), torch.from_numpy(mask))


# --------------------------------------------------------------------------- #
# Loaders (Rust .npy) + Python fallback
# --------------------------------------------------------------------------- #
def load_hypergraph(path: Path | str) -> Hypergraph:
    """Load a Rust-preprocessed hypergraph from ``<path>/`` containing
    ``assignment.npy`` and ``edge_lists.npy``."""
    path = Path(path)
    assign = np.load(path / "assignment.npy").reshape(-1).astype(np.int64)
    n_nodes = assign.shape[0]
    n_edges = int(assign.max()) + 1 if n_nodes > 0 else 0
    edge_lists: List[List[int]] = [[] for _ in range(n_edges)]
    for node, e in enumerate(assign.tolist()):
        edge_lists[e].append(node)
    for el in edge_lists:
        el.sort()
    return Hypergraph(
        n_nodes=n_nodes, n_edges=n_edges,
        assignment=torch.from_numpy(assign),
        edge_lists=edge_lists,
    )


def load_dilated_structure(path: Path | str, levels: Tuple[int, ...],
                           n_nodes: int, max_neighbors: int = 32,
                           seed: int = 42) -> DilatedStructure:
    """Load Rust-exported dilated neighbor CSR arrays for each level."""
    path = Path(path)
    padded, mask = {}, {}
    for k in levels:
        nb = np.load(path / f"dil_neighbors_{k}.npy").reshape(-1).astype(np.int64)
        off = np.load(path / f"dil_offsets_{k}.npy").reshape(-1).astype(np.int64)
        p, m = _csr_to_padded(nb, off, n_nodes, max_neighbors, seed=seed + k)
        padded[k] = p
        mask[k] = m
    return DilatedStructure(levels=levels, padded=padded, mask=mask)


def build_hypergraph_from_features(feats: torch.Tensor, k: Optional[int] = None,
                                    seed: int = 42) -> Hypergraph:
    """Pure-Python cosine-similarity k-means clustering fallback.

    Mirrors ``rust/src/clustering.rs::cosine_clusters`` so the pipeline runs
    without invoking the Rust preprocessor. Features are L2-normalized and
    clustered with k-means++ init + Lloyd iterations using cosine similarity.
    """
    n = feats.shape[0]
    if k is None:
        k = max(8, int(n ** 0.5))
    k = min(k, n)
    device = feats.device

    x = torch.nn.functional.normalize(feats, dim=-1)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    # k-means++ init
    first = int(torch.randint(0, n, (1,), generator=gen).item())
    centroids = [x[first].cpu()]
    dist2 = torch.full((n,), float("inf"))
    for _ in range(1, k):
        c = centroids[-1].to(device)
        sim = x @ c
        d2 = (1.0 - sim).clamp_min(0.0)
        dist2 = torch.minimum(dist2, d2)
        probs = dist2 / dist2.sum().clamp_min(1e-12)
        pick = int(torch.multinomial(probs.cpu(), 1, generator=gen).item())
        centroids.append(x[pick].cpu())
    centroids = torch.stack(centroids).to(device)  # (k, D)

    # Lloyd iterations
    assignment = torch.zeros(n, dtype=torch.long, device=device)
    for _ in range(50):
        # cosine sim = dot (normalized)
        sims = x @ centroids.t()           # (N, k)
        new_assign = sims.argmax(dim=-1)
        if torch.equal(new_assign, assignment):
            break
        assignment = new_assign
        for c in range(k):
            members = x[assignment == c]
            if members.shape[0] > 0:
                centroids[c] = torch.nn.functional.normalize(
                    members.mean(0), dim=-1)
            else:
                # reseed dead centroid
                r = int(torch.randint(0, n, (1,), generator=gen).item())
                centroids[c] = x[r]

    edge_lists: List[List[int]] = [[] for _ in range(k)]
    for node, e in enumerate(assignment.tolist()):
        edge_lists[e].append(node)
    return Hypergraph(
        n_nodes=n, n_edges=k, assignment=assignment, edge_lists=edge_lists,
    )
