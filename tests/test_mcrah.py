"""Tests for the MCRAH novel module (motion-coherent rigidity-adaptive hypergraph).

These run on CPU and verify:
  1. The gate produces a valid soft membership (rows sum to 1, in [0,1]).
  2. Soft propagation through the adaptive hypergraph preserves shape + scale.
  3. Rigidity loss penalizes intra-cluster deformation deviation.
  4. Topology smoothness loss penalizes membership jumps.
  5. The full MCRAH model with MCRAHCore enabled produces a valid rollout with membership.
  6. MCRAHCore disabled (static) path still works (ablation arm).
  7. Gradients flow through the membership gate to the gate parameters.

Run: pytest tests/test_mcrah.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from mcrah.config import Config
from mcrah.gs.gaussian import GaussianCloud
from mcrah.models.hypergraph import build_hypergraph_from_features
from mcrah.models.mcrah import (
    MCRAHCore, MCRAHGate, AdaptiveHypergraph,
    RigidityLoss, TopologySmoothnessLoss, compute_centroids,
)
from mcrah.models.network import MCRAH


@pytest.fixture
def tiny_cloud():
    torch.manual_seed(0)
    n = 64
    return GaussianCloud(
        means=torch.randn(n, 3) * 0.3,
        scales=torch.full((n, 3), -2.0),
        rotations=torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(n, 4).contiguous(),
        opacities=torch.zeros(n, 1),
        sh=torch.rand(n, 3),
    )


@pytest.fixture
def cfg():
    c = Config()
    c.model.feat_dim = 32
    c.model.hidden_dim = 32
    c.model.num_heads = 4
    c.model.num_simgnn_layers = 2
    c.model.num_dhgc_layers = 1
    c.model.noise_std = 0.0
    c.hypergraph.adaptive = True
    c.hypergraph.mcrae_tau = 1.0
    c.train.device = "cpu"
    return c


def test_gate_produces_valid_membership(tiny_cloud):
    """Gate output M is (N, E), rows sum to 1, all in [0, 1]."""
    hg = build_hypergraph_from_features(tiny_cloud.means, k=8, seed=0)
    centroids_0 = compute_centroids(tiny_cloud.means, hg.assignment, hg.n_edges)
    gate = MCRAHGate(feat_dim=32, n_clusters=hg.n_edges, tau=1.0)
    gate.eval()

    M = gate(tiny_cloud.means, tiny_cloud.means, hg.assignment, centroids_0)
    assert M.shape == (tiny_cloud.n, hg.n_edges)
    assert torch.allclose(M.sum(dim=-1), torch.ones(tiny_cloud.n), atol=1e-5)
    assert (M >= 0).all() and (M <= 1).all()


def test_adaptive_hypergraph_propagation(tiny_cloud):
    """Soft propagation preserves shape and is finite."""
    hg = build_hypergraph_from_features(tiny_cloud.means, k=8, seed=0)
    centroids_0 = compute_centroids(tiny_cloud.means, hg.assignment, hg.n_edges)
    mcrah = MCRAHCore(hg.assignment, hg.n_edges, feat_dim=32, tau=1.0, adaptive=True)
    mcrah.set_reference(tiny_cloud.means, centroids_0)
    mcrah.update_membership(tiny_cloud.means)

    x = torch.randn(tiny_cloud.n, 16)
    y = mcrah.propagate(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    # Soft Θ is normalized; output should be on the same order as input.
    assert y.norm() <= x.norm() * 1.5 + 1e-6


def test_rigidity_loss_penalizes_deformation(tiny_cloud):
    """Rigidity loss is larger when nodes deviate from cluster-centroid motion."""
    hg = build_hypergraph_from_features(tiny_cloud.means, k=8, seed=0)
    centroids_0 = compute_centroids(tiny_cloud.means, hg.assignment, hg.n_edges)
    loss_fn = RigidityLoss()

    # One-hot membership (hard assignment).
    M = torch.nn.functional.one_hot(
        hg.assignment, num_classes=hg.n_edges).float()

    # Rigid translation: all nodes move by the same amount => low loss.
    means_rigid = tiny_cloud.means + torch.tensor([0.1, 0.0, 0.0])
    loss_rigid = loss_fn(means_rigid, tiny_cloud.means, M, hg.assignment)

    # Non-rigid: each node moves randomly => high loss.
    means_nonrigid = tiny_cloud.means + torch.randn_like(tiny_cloud.means) * 0.1
    loss_nonrigid = loss_fn(means_nonrigid, tiny_cloud.means, M, hg.assignment)

    assert loss_nonrigid > loss_rigid


def test_topology_smoothness_loss(tiny_cloud):
    """Topology loss is zero for identical memberships, positive for different."""
    hg = build_hypergraph_from_features(tiny_cloud.means, k=8, seed=0)
    loss_fn = TopologySmoothnessLoss()
    M = torch.nn.functional.one_hot(
        hg.assignment, num_classes=hg.n_edges).float()
    assert loss_fn(M, M).item() < 1e-8

    M2 = M.clone()
    # Swap two rows' assignments.
    M2[0], M2[1] = M[1].clone(), M[0].clone()
    assert loss_fn(M2, M).item() > 0


def test_mcrah_rollout_with_mcrah(tiny_cloud, cfg):
    """Full rollout with MCRAH produces valid steps with membership."""
    model = MCRAH(cfg, tiny_cloud)
    model.eval()
    assert model.use_mcrah is True
    assert model.mcrah is not None

    times = [torch.tensor(0.0), torch.tensor(0.2), torch.tensor(0.4)]
    steps = model.rollout(times)
    assert len(steps) == 3
    for s in steps:
        assert s.cloud.n == tiny_cloud.n
        assert s.delta_pos.shape == (tiny_cloud.n, 3)
        assert torch.isfinite(s.cloud.means).all()
        assert s.membership is not None
        assert s.membership.shape[0] == tiny_cloud.n
        assert torch.allclose(
            s.membership.sum(dim=-1), torch.ones(tiny_cloud.n), atol=1e-5)


def test_mcrah_rollout_without_mcrah(tiny_cloud, cfg):
    """Static (ablation) path still works."""
    cfg.hypergraph.adaptive = False
    model = MCRAH(cfg, tiny_cloud)
    model.eval()
    assert model.use_mcrah is False
    assert model.mcrah is None

    times = [torch.tensor(0.0), torch.tensor(0.2)]
    steps = model.rollout(times)
    assert len(steps) == 2
    for s in steps:
        assert s.membership is None
        assert torch.isfinite(s.cloud.means).all()


def test_mcrah_gradient_flows_through_gate(tiny_cloud, cfg):
    """Gradients flow from the rigidity loss back through the membership gate."""
    model = MCRAH(cfg, tiny_cloud)
    model.train()

    hg = model.hypergraph
    centroids_0 = compute_centroids(tiny_cloud.means, hg.assignment, hg.n_edges)
    model.mcrah.set_reference(tiny_cloud.means, centroids_0)

    times = [torch.tensor(0.0), torch.tensor(0.1)]
    steps = model.rollout(times)

    rigidity = RigidityLoss()
    loss = sum(
        rigidity(s.cloud.means, model.mcrah.means_0,
                 s.membership, model.mcrah.hypergraph.assignment_0)
        for s in steps if s.membership is not None
    ) / len(steps)
    loss.backward()

    # Gate parameters must have gradients.
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.mcrah.gate.parameters()
    )
    assert has_grad, "No gradient reached the MCRAH gate parameters"
