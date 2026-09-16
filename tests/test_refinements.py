"""Unit tests for MCRAH architectural refinements & training pipeline upgrades.

Tests:
  1. FourierEmbedding generates correct dimensions and finite multi-frequency bands.
  2. FeatureEncoder integrates Fourier features for 3D coordinates and time.
  3. OffsetHeads modulates outputs with residual motion damping and pos_scale.
  4. MCRAHTrainer initializes with stage-specific learning rates and decays via cosine annealing.
  5. MCRAHTrainer activates LPIPS in joint stage and records perceptual metrics.
  6. StaticGSInit decays parameter groups properly without premature freeze.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from mcrah.config import Config
from mcrah.data import SceneSample
from mcrah.gs.gaussian import GaussianCloud
from mcrah.models.hypergraph import build_hypergraph_from_features
from mcrah.models.simgnn import FeatureEncoder, FourierEmbedding, OffsetHeads, SIMGNN
from mcrah.training import Evaluator, MCRAHTrainer, StaticGSInit


@pytest.fixture
def tiny_cloud():
    torch.manual_seed(42)
    n = 32
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
    c.model.num_freqs_pos = 6
    c.model.num_freqs_time = 4
    c.model.pos_scale = 0.20
    c.model.motion_damping = 0.90
    c.train.device = "cpu"
    c.train.iterations = 20
    c.train.time_window = 2
    c.train.batch_views = 2
    c.train.w_lpips = 0.05
    return c


def test_fourier_embedding():
    """FourierEmbedding transforms inputs with sinusoidal multi-frequency bands."""
    embed_pos = FourierEmbedding(in_dim=3, num_freqs=6, include_input=True)
    assert embed_pos.out_dim == 3 + 3 * 2 * 6  # 39
    x = torch.randn(10, 3)
    out = embed_pos(x)
    assert out.shape == (10, 39)
    assert torch.isfinite(out).all()
    # First 3 dimensions should equal original x
    assert torch.allclose(out[:, :3], x)

    embed_time = FourierEmbedding(in_dim=1, num_freqs=4, include_input=True)
    assert embed_time.out_dim == 1 + 1 * 2 * 4  # 9
    t = torch.tensor([[0.5], [0.8]])
    out_t = embed_time(t)
    assert out_t.shape == (2, 9)
    assert torch.isfinite(out_t).all()


def test_feature_encoder_with_fourier(tiny_cloud):
    """FeatureEncoder integrates Fourier coordinate and temporal features."""
    enc = FeatureEncoder(
        feat_dim=32, n_clusters=16, num_freqs_pos=6, num_freqs_time=4
    )
    cluster_id = torch.randint(0, 16, (tiny_cloud.n,))
    t = torch.tensor(0.25)
    h = enc(tiny_cloud, cluster_id, t)
    assert h.shape == (tiny_cloud.n, 32)
    assert torch.isfinite(h).all()


def test_offset_heads_residual_damping():
    """OffsetHeads respects pos_scale and residual motion damping."""
    heads = OffsetHeads(
        dim=32, predict_rotation=True, pos_scale=0.20, motion_damping=0.90
    )
    heads.eval()
    h = torch.randn(16, 32)
    dp, dr = heads(h)
    assert dp.shape == (16, 3)
    assert dr.shape == (16, 4)
    # Positions must be bounded by pos_scale
    assert (dp.abs() <= 0.20 + 1e-5).all()
    # Rotations must be unit quaternions
    assert torch.allclose(dr.norm(dim=-1), torch.ones(16), atol=1e-4)


def test_trainer_stage_learning_rates_and_scheduler(tiny_cloud, cfg):
    """MCRAHTrainer switches stage-specific LRs and applies CosineAnnealingLR."""
    hg = build_hypergraph_from_features(tiny_cloud.means.detach(), k=4, seed=0)
    trainer = MCRAHTrainer(cfg, tiny_cloud, hypergraph=hg, out_dir="/tmp/mcrah_test_sched")

    # Dense stage
    trainer.set_stage("dense", total_steps=50)
    assert trainer.state.stage == "dense"
    assert trainer._opt.param_groups[0]["lr"] == cfg.train.lr_dense

    # Farfield stage
    trainer.set_stage("farfield", total_steps=50)
    assert trainer.state.stage == "farfield"
    assert trainer._opt.param_groups[0]["lr"] == cfg.train.lr_farfield

    # Joint stage
    trainer.set_stage("joint", total_steps=50)
    assert trainer.state.stage == "joint"
    assert trainer._opt.param_groups[0]["lr"] == cfg.train.lr_joint

    # Execute a step and verify scheduler decay
    K = torch.tensor([[400.0, 0.0, 100.0], [0.0, 400.0, 100.0], [0.0, 0.0, 1.0]])
    c2w = torch.eye(4); c2w[2, 3] = -4.0
    img = torch.rand(3, 32, 32)
    samples = [
        SceneSample(category="x", time=0.0, time_idx=0, image=img, c2w=c2w, intrinsics=K),
        SceneSample(category="x", time=0.1, time_idx=1, image=img, c2w=c2w, intrinsics=K),
    ]
    m1 = trainer.train_step(samples)
    assert "lr" in m1
    m2 = trainer.train_step(samples)
    assert m2["lr"] <= m1["lr"]


def test_trainer_lpips_in_joint_stage(tiny_cloud, cfg):
    """Joint stage computes perceptual LPIPS loss when w_lpips > 0."""
    hg = build_hypergraph_from_features(tiny_cloud.means.detach(), k=4, seed=0)
    trainer = MCRAHTrainer(cfg, tiny_cloud, hypergraph=hg, out_dir="/tmp/mcrah_test_lpips")
    trainer.set_stage("joint")

    K = torch.tensor([[400.0, 0.0, 100.0], [0.0, 400.0, 100.0], [0.0, 0.0, 1.0]])
    c2w = torch.eye(4); c2w[2, 3] = -4.0
    img = torch.rand(3, 32, 32)
    samples = [
        SceneSample(category="x", time=0.0, time_idx=0, image=img, c2w=c2w, intrinsics=K),
        SceneSample(category="x", time=0.1, time_idx=1, image=img, c2w=c2w, intrinsics=K),
    ]
    m = trainer.train_step(samples)
    assert "lpips" in m
    assert m["lpips"] >= 0.0


def test_static_init_decoupled_schedule():
    """StaticGSInit with decoupled parameter scheduling steadily reduces loss."""
    torch.manual_seed(0)
    cfg = Config()
    cfg.train.device = "cpu"
    cfg.static_gs.num_gaussians = 32

    cloud = GaussianCloud.random(32)
    K = torch.tensor([[400.0, 0.0, 100.0], [0.0, 400.0, 100.0], [0.0, 0.0, 1.0]])
    c2w = torch.eye(4); c2w[2, 3] = -4.0
    from mcrah.gs import set_rasterizer, render
    set_rasterizer("torch")
    target = render(cloud.activated(), c2w, K, width=32, height=32).image

    init = StaticGSInit(cfg, device="cpu")
    views = [(target, c2w, K)]
    result = init.fit(views, iterations=12)
    assert len(result.history) == 12
    # Final loss should be lower than starting loss
    assert result.history[-1] < result.history[0]
