"""Smoke tests for the MCRAH pipeline on a tiny synthetic scene.

These run on CPU (no CUDA/MPS dependency) and verify:
  1. The package imports cleanly (regression test for the missing rasterizer).
  2. The hypergraph propagation operator is sane (Θ preserves feature scale).
  3. SIMGNN predicts correctly-shaped offsets.
  4. DHGC restricted attention runs and respects masks.
  5. The full MCRAH autoregressive rollout produces a valid cloud sequence.
  6. The pure-torch rasterizer renders a non-empty image from a Gaussian.

Run: pytest tests/test_smoke.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure src/ is importable without an install (also via pyproject pythonpath).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from mcrah.config import Config
from mcrah.gs.gaussian import GaussianCloud, apply_offsets
from mcrah.gs import render, set_rasterizer, PyTorchRasterizer
from mcrah.models.hypergraph import build_hypergraph_from_features
from mcrah.models.simgnn import SIMGNN
from mcrah.models.dhgc import DHGC, DilatedAttention
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
    c.train.device = "cpu"
    return c


def test_package_imports():
    """Regression: gs/__init__.py must not reference a missing module."""
    import mcrah
    import mcrah.gs
    import mcrah.models
    import mcrah.losses
    assert mcrah.__version__ == "0.1.0"


def test_hypergraph_propagation_preserves_scale(tiny_cloud):
    hg = build_hypergraph_from_features(tiny_cloud.means, k=8, seed=0)
    x = torch.randn(hg.n_nodes, 16)
    y = hg.propagate(x)
    # Θ is a normalized operator; output magnitude should be on the same order.
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert y.norm() <= x.norm() * 1.5 + 1e-6


def test_simgnn_offset_shapes(tiny_cloud, cfg):
    hg = build_hypergraph_from_features(tiny_cloud.means, k=8, seed=0)
    net = SIMGNN(cfg)
    net.eval()
    t = torch.tensor(0.1)
    dp, dr = net(tiny_cloud, hg, hg.assignment, t)
    assert dp.shape == (tiny_cloud.n, 3)
    assert dr.shape == (tiny_cloud.n, 4)
    assert torch.allclose(dr.norm(dim=-1), torch.ones(tiny_cloud.n), atol=1e-4)


def test_dhgc_attention_respects_mask(tiny_cloud, cfg):
    # Build a trivial dilation structure: every node attends to 2 fixed neighbors.
    N = tiny_cloud.n
    max_deg = 2
    idx = torch.zeros(N, max_deg, dtype=torch.long)
    mask = torch.zeros(N, max_deg, dtype=torch.bool)
    for i in range(N):
        idx[i, 0] = (i + 1) % N
        mask[i, 0] = True
        idx[i, 1] = (i + 2) % N
        mask[i, 1] = True

    attn = DilatedAttention(cfg.model.feat_dim, cfg.model.num_heads)
    attn.eval()
    x = torch.randn(N, cfg.model.feat_dim)
    out = attn(x, idx, mask)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    # A zeroed mask must not produce NaNs (all -inf -> uniform softmax).
    bad_mask = torch.zeros(N, max_deg, dtype=torch.bool)
    out2 = attn(x, idx, bad_mask)
    assert torch.isfinite(out2).all()


def test_mcrah_rollout(tiny_cloud, cfg):
    model = MCRAH(cfg, tiny_cloud)
    model.eval()
    times = [torch.tensor(0.0), torch.tensor(0.2), torch.tensor(0.4)]
    steps = model.rollout(times)
    assert len(steps) == 3
    for s in steps:
        assert s.cloud.n == tiny_cloud.n
        assert s.delta_pos.shape == (tiny_cloud.n, 3)
        assert torch.isfinite(s.cloud.means).all()


def test_mcrah_stage_freezing(tiny_cloud, cfg):
    model = MCRAH(cfg, tiny_cloud)
    model.configure_stage("dense")
    assert all(not p.requires_grad for p in model.dhgc.parameters())
    assert any(p.requires_grad for p in model.simgnn.parameters())

    model.configure_stage("farfield")
    assert all(not p.requires_grad for p in model.simgnn.parameters())
    assert any(p.requires_grad for p in model.dhgc.parameters())

    model.configure_stage("joint")
    assert any(p.requires_grad for p in model.simgnn.parameters())
    assert any(p.requires_grad for p in model.dhgc.parameters())


def test_pytorch_rasterizer_renders(tiny_cloud):
    set_rasterizer("torch")
    K = torch.tensor([[400.0, 0.0, 100.0], [0.0, 400.0, 100.0],
                      [0.0, 0.0, 1.0]])
    # OpenCV convention (used throughout scene.py): camera looks down +Z, so to
    # view the cloud at the origin the camera must sit on the -Z side at z=-4.
    c2w = torch.eye(4)
    c2w[2, 3] = -4.0
    out = render(tiny_cloud, c2w, K, width=200, height=200)
    assert out.image.shape == (3, 200, 200)
    assert out.alpha.shape == (1, 200, 200)
    assert out.image.min() >= 0.0 and out.image.max() <= 1.0
    assert out.n_visible > 0
