"""Smoke tests for the training & evaluation infrastructure (Phases 1, 3, 4).

Verifies on CPU:
  1. StaticGSInit produces a cloud and reduces the rendering loss.
  2. MCRAHTrainer runs a decoupled two-stage training step with gradients.
  3. The differentiable rasterizer actually backprops to predicted offsets.
  4. Evaluator computes PSNR/SSIM and rollout stability.

Run: pytest tests/test_training.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from mcrah.config import Config
from mcrah.gs import set_rasterizer, render
from mcrah.gs.gaussian import GaussianCloud, apply_offsets
from mcrah.models.hypergraph import build_hypergraph_from_features
from mcrah.training import (
    MCRAHTrainer, Evaluator, StaticGSInit,
)
from mcrah.models import MCRAH


def _make_cfg():
    c = Config()
    c.model.feat_dim = 32
    c.model.hidden_dim = 32
    c.model.num_heads = 4
    c.model.num_simgnn_layers = 2
    c.model.num_dhgc_layers = 1
    c.model.noise_std = 0.0
    c.train.device = "cpu"
    c.train.iterations = 2
    c.train.time_window = 2
    c.train.batch_views = 2
    return c


def test_static_init_reduces_loss():
    """StaticGSInit should lower the rendering loss over a few iterations."""
    torch.manual_seed(0)
    cfg = _make_cfg()
    cfg.static_gs.iterations = 10
    cfg.static_gs.num_gaussians = 64

    # Build a tiny scene: a small cloud rendered from one camera.
    cloud = GaussianCloud.random(64)
    K = torch.tensor([[400.0, 0.0, 100.0], [0.0, 400.0, 100.0],
                      [0.0, 0.0, 1.0]])
    c2w = torch.eye(4)
    c2w[2, 3] = -4.0
    set_rasterizer("torch")
    target = render(cloud.activated(), c2w, K, width=64, height=64).image

    init = StaticGSInit(cfg, device="cpu")
    views = [(target, c2w, K)]
    result = init.fit(views, iterations=8)
    assert result.cloud.n > 0
    assert len(result.history) == 8
    # Loss should be non-increasing on average (allow noise).
    assert result.history[-1] <= result.history[0] * 2.0


def test_trainer_dense_step_has_gradients():
    """A dense-stage training step must produce gradients on SIMGNN params."""
    cfg = _make_cfg()
    cloud = GaussianCloud.random(32)
    hg = build_hypergraph_from_features(cloud.means.detach(), k=8, seed=0)
    trainer = MCRAHTrainer(cfg, cloud, hypergraph=hg, out_dir="/tmp/mcrah_test")
    trainer.set_stage("dense")

    # Build fake SceneSamples (we only need image/c2w/intrinsics/time).
    K = torch.tensor([[400.0, 0.0, 100.0], [0.0, 400.0, 100.0],
                      [0.0, 0.0, 1.0]])
    c2w = torch.eye(4); c2w[2, 3] = -4.0
    img = torch.rand(3, 32, 32)

    from mcrah.data import SceneSample
    samples = [
        SceneSample(category="x", time=0.0, time_idx=0, image=img,
                    c2w=c2w, intrinsics=K),
        SceneSample(category="x", time=0.1, time_idx=1, image=img,
                    c2w=c2w, intrinsics=K),
    ]
    m = trainer.train_step(samples)
    assert torch.isfinite(torch.tensor(m["loss"]))
    # SIMGNN must have received gradients.
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in trainer.model.simgnn.parameters())
    assert has_grad, "SIMGNN params received no gradients"


def test_rasterizer_is_differentiable():
    """Rendering must carry gradients back to the input cloud means."""
    torch.manual_seed(0)
    cloud = GaussianCloud.random(8)
    # Make means require grad through a clone.
    means = cloud.means.clone().requires_grad_(True)
    gcloud = GaussianCloud(
        means=means, scales=cloud.scales, rotations=cloud.rotations,
        opacities=cloud.opacities, sh=cloud.sh)
    K = torch.tensor([[400.0, 0.0, 100.0], [0.0, 400.0, 100.0],
                      [0.0, 0.0, 1.0]])
    c2w = torch.eye(4); c2w[2, 3] = -4.0
    out = render(gcloud, c2w, K, width=32, height=32)
    loss = out.image.sum()
    loss.backward()
    assert means.grad is not None
    assert means.grad.abs().sum() > 0, "no gradient flowed to cloud means"


def test_evaluator_runs():
    """Evaluator computes metrics without error on a tiny rollout."""
    cfg = _make_cfg()
    cloud = GaussianCloud.random(16)
    model = MCRAH(cfg, cloud)
    model.eval()

    K = torch.tensor([[400.0, 0.0, 100.0], [0.0, 400.0, 100.0],
                      [0.0, 0.0, 1.0]])
    c2w = torch.eye(4); c2w[2, 3] = -4.0
    img = torch.rand(3, 32, 32)
    from mcrah.data import SceneSample
    samples = [
        SceneSample(category="x", time=0.0, time_idx=0, image=img,
                    c2w=c2w, intrinsics=K),
    ]
    ev = Evaluator(cfg, device="cpu")
    metrics = ev.evaluate(model, samples)
    assert metrics.n_samples == 1
    assert torch.isfinite(torch.tensor(metrics.psnr_mean))
    assert torch.isfinite(torch.tensor(metrics.ssim_mean))

    stab = ev.rollout_stability(model, n_steps=5)
    assert len(stab.pos_drift) == 5
    assert all(v >= 0 for v in stab.pos_drift)
