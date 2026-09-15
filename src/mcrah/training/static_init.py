"""Phase 1 Step 3: Static 3D Gaussian Splatting initialization.

For each D-NeRF category we isolate the $t=0$ frame(s) and fit a *static* 3DGS
model to a single time step. The resulting cloud (means, scales, rotations,
opacities, sh) is the deformation substrate the MCRAH autoregresses over
(workflow.md Phase 3 Step 7): ``cloud_{t+1} = apply_offsets(cloud_t, Δpos, Δrot)``.

This is a lightweight point-based initializer rather than the full 3DGS density
control + adaptive bound training: it seeds Gaussians from the scene depth and
optimizes their attributes by directly minimizing the L1+SSIM rendering loss on
the $t=0$ cameras using the pure-torch differentiable rasterizer. It runs on
Apple Silicon / MPS and produces a good-enough substrate for network training;
a production run would substitute the full CUDA 3DGS optimization here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from ..gs import render, set_rasterizer
from ..gs.gaussian import GaussianCloud, quaternion_normalize
from ..losses import L1SSIMLoss


@dataclass
class StaticInitResult:
    cloud: GaussianCloud
    history: List[float]
    iterations: int


def seed_cloud_from_points(
    means: torch.Tensor,
    device: torch.device,
    sh_degree: int = 0,
) -> GaussianCloud:
    """Build a GaussianCloud with sensible defaults from initial 3D points.

    Scales are set so each splat covers a small neighborhood; opacities start
    at ~0.1 (sigmoid(0) = 0.5 -> we use logit 0); rotations are identity.
    """
    n = means.shape[0]
    # Estimate per-point scale from the scene extent, not nearest-neighbor
    # distance.  Using torch.cdist on N=50k points allocates an N×N matrix
    # (~10 GB) and the resulting NN distances are so tiny that the splats
    # are sub-pixel — the rendered image is piecewise-constant w.r.t. means,
    # giving zero gradient.  Instead use a fraction of the scene diagonal so
    # each splat covers a few pixels and the rendering loss has gradient.
    if n > 1:
        scene_diag = (means.max(dim=0).values - means.min(dim=0).values).norm()
        scene_diag = scene_diag.clamp_min(1e-3)
        # Each splat should be roughly scene_diag / sqrt(N) wide — small enough
        # to be local, large enough to cover >1 pixel at the render resolution.
        splat_size = scene_diag / (n ** 0.5 + 1e-6)
        log_scale = torch.full((n, 3), float(torch.log(splat_size)), device=device)
    else:
        log_scale = torch.full((n, 3), -2.0)
    # SH: 1 coefficient (DC) when degree 0. Initialize to a nonzero mid-gray
    # (0.5) so the rendered image is NOT constant-black. A zero-init sh makes
    # every Gaussian black, so the rendered image is identical regardless of
    # position/opacity/scale, giving zero gradient and a completely flat loss.
    sh = torch.full((n, 3), 0.5, device=device)
    return GaussianCloud(
        means=means.to(device).detach().clone().requires_grad_(True),
        scales=log_scale.to(device).detach().clone().requires_grad_(True),
        rotations=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        .expand(n, 4).contiguous().detach().clone().requires_grad_(True),
        opacities=torch.zeros(n, 1, device=device).detach().clone()
        .requires_grad_(True),
        sh=sh.detach().clone().requires_grad_(True),
    )


def points_from_depth(
    c2w: torch.Tensor,
    intrinsics: torch.Tensor,
    depth: torch.Tensor,
    n_samples: int,
    min_depth: float = 0.5,
    max_depth: float = 6.0,
) -> torch.Tensor:
    """Back-project depth pixels to 3D points (OpenCV camera convention).

    depth: (H, W) float; samples ``n_samples`` pixels whose depth is in range.
    Returns (n_samples, 3) world-space points.
    """
    H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    valid = (depth > min_depth) & (depth < max_depth)
    ys, xs = torch.meshgrid(
        torch.arange(H, device=depth.device),
        torch.arange(W, device=depth.device), indexing="ij")
    z = depth[valid]
    if z.numel() == 0:
        return torch.zeros(0, 3, device=depth.device)
    n_samples = min(n_samples, z.numel())
    idx = torch.randperm(z.numel(), device=depth.device)[:n_samples]
    z = z[idx]
    x = xs[valid][idx]
    y = ys[valid][idx]
    pts_cam = torch.stack([
        (x - cx) * z / fx, (y - cy) * z / fy, z,
    ], dim=-1)  # (S,3)
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    return pts_cam @ R.T + t  # (S,3) world


def seed_cloud_from_views(
    views: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    n_gaussians: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Seed 3D points inside the 3D visual hull of the object centered at (0,0,0).

    Generates candidate 3D points in [-0.45, 0.45]^3 world space, projects them into
    all t=0 camera views, and selects points that project to foreground pixels across
    multiple views with initial colors sampled from projected 2D RGB pixels.

    Returns (points (N,3), colors (N,3), initial_opacities (N,1)).
    """
    # 1. Generate candidate points in world space [-0.45, 0.45]^3 around origin
    n_candidates = max(n_gaussians * 3, 100_000)
    candidates = (torch.rand(n_candidates, 3, device=device) - 0.5) * 0.9  # [-0.45, 0.45]^3

    accum_color = torch.zeros(n_candidates, 3, device=device)
    hit_count = torch.zeros(n_candidates, device=device)

    # 2. Project candidates into each camera view
    for img, c2w, K in views:
        H, W = img.shape[-2], img.shape[-1]
        w2c = torch.inverse(c2w)

        R_w2c = w2c[:3, :3]
        t_w2c = w2c[:3, 3]
        pts_cam = candidates @ R_w2c.T + t_w2c  # (N, 3)

        z = pts_cam[:, 2]
        in_front = z > 0.1

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        u = (fx * pts_cam[:, 0] / z + cx).round().long()
        v = (fy * pts_cam[:, 1] / z + cy).round().long()

        in_bounds = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)

        if not in_bounds.any():
            continue

        valid_idx = torch.where(in_bounds)[0]
        u_v = u[valid_idx]
        v_v = v[valid_idx]

        pix_colors = img[:, v_v, u_v].t()  # (M, 3)
        is_fg = (pix_colors < 0.92).any(dim=-1)  # (M,) bool

        fg_valid_idx = valid_idx[is_fg]
        fg_pix_colors = pix_colors[is_fg]

        accum_color.index_add_(0, fg_valid_idx, fg_pix_colors)
        hit_count.index_add_(0, fg_valid_idx, torch.ones_like(fg_valid_idx, dtype=torch.float))

    # 3. Select points inside visual hull (hit_count > 0)
    fg_mask = hit_count > 0
    if fg_mask.sum() > 100:
        hull_idx = torch.where(fg_mask)[0]
        sorted_idx = hull_idx[torch.argsort(hit_count[hull_idx], descending=True)]

        if len(sorted_idx) >= n_gaussians:
            sel_idx = sorted_idx[:n_gaussians]
        else:
            repeat_cnt = (n_gaussians // len(sorted_idx)) + 1
            sel_idx = sorted_idx.repeat(repeat_cnt)[:n_gaussians]

        selected_pts = candidates[sel_idx] + torch.randn(n_gaussians, 3, device=device) * 0.005
        selected_colors = accum_color[sel_idx] / hit_count[sel_idx].unsqueeze(-1).clamp_min(1.0)
        selected_opacities = torch.full((n_gaussians, 1), 0.5, device=device)
        return selected_pts, selected_colors, selected_opacities
    else:
        pts = torch.randn(n_gaussians, 3, device=device) * 0.35
        colors = torch.full((n_gaussians, 3), 0.3, device=device)
        opacities = torch.zeros(n_gaussians, 1, device=device)
        return pts, colors, opacities


class StaticGSInit:
    """Optimize a static Gaussian cloud against t=0 cameras.

    Usage::

        init = StaticGSInit(cfg, device)
        result = init.fit(t0_views)
        cloud = result.cloud
    """

    def __init__(self, cfg: Config, device: Optional[str] = None):
        self.cfg = cfg
        self.device = device or cfg.device_str()
        if self.device == "cpu":
            set_rasterizer("torch")
        else:
            set_rasterizer("auto")
        self.loss_fn = L1SSIMLoss(
            w_l1=cfg.train.w_l1, w_ssim=cfg.train.w_ssim)

    def fit(
        self,
        views: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        init_points: Optional[torch.Tensor] = None,
        iterations: Optional[int] = None,
    ) -> StaticInitResult:
        """Fit a static cloud to ``views`` = list of (image, c2w, intrinsics).

        ``init_points``: optional (M,3) seed points; if None, points are
        seeded via 3D visual hull projection.
        """
        cfg = self.cfg
        iters = iterations or cfg.static_gs.iterations
        dev = self.device

        # Move views to device.
        rH, rW = cfg.data.render_wh[1], cfg.data.render_wh[0]
        views = [
            (F.interpolate(img.unsqueeze(0), size=(rH, rW), mode="bilinear",
                           align_corners=False).squeeze(0).to(dev),
             c2w.to(dev), K.to(dev))
            for img, c2w, K in views
        ]
        H, W = views[0][0].shape[-2], views[0][0].shape[-1]

        # Seed points & colors from 3D visual hull projection
        if init_points is None:
            n = cfg.static_gs.num_gaussians
            init_points, init_colors, init_opacities = seed_cloud_from_views(views, n, dev)
        else:
            init_colors = None
            init_opacities = None

        cloud = seed_cloud_from_points(init_points, dev)

        # Initialize SH colors, opacities, and scales
        with torch.no_grad():
            if init_colors is not None:
                cloud.sh.data.copy_(init_colors)
            if init_opacities is not None:
                cloud.opacities.data.copy_(init_opacities)
            else:
                cloud.opacities.data.fill_(-0.5)
            cloud.scales.data.clamp_(-6.0, -2.8)

        # Parameter groups with per-attribute learning rates (3DGS convention).
        params = [
            {"params": [cloud.means], "lr": cfg.static_gs.lr_means * 1.5},
            {"params": [cloud.scales], "lr": cfg.static_gs.lr_scales},
            {"params": [cloud.opacities], "lr": cfg.static_gs.lr_opacity},
            {"params": [cloud.sh], "lr": 3.5e-3},
        ]
        if cfg.model.predict_rotation:
            params.append(
                {"params": [cloud.rotations], "lr": cfg.static_gs.lr_means})
        opt = torch.optim.Adam(params, lr=cfg.static_gs.lr_means)
        sched = torch.optim.lr_scheduler.ExponentialLR(
            opt, gamma=0.992)

        history: List[float] = []
        bg = torch.ones(3, device=dev) if cfg.data.white_background else None
        for it in range(iters):
            opt.zero_grad()
            total = 0.0
            for img, c2w, K in views:
                out = render(cloud, c2w, K, width=W, height=H, bg_color=bg)
                pred = out.image.unsqueeze(0)  # (1,3,H,W)
                tgt = img.unsqueeze(0)
                loss = self.loss_fn(pred, tgt)
                total = total + loss
            total = total / len(views)
            total.backward()
            opt.step()
            sched.step()

            # Keep parameters physically bounded during optimization
            with torch.no_grad():
                cloud.scales.data.clamp_(-6.0, -2.8)
                cloud.sh.data.clamp_(0.0, 1.0)
                cloud.opacities.data.clamp_(-3.5, 4.0)

            history.append(float(total.item()))
            if (it + 1) % max(1, iters // 10) == 0:
                print(f"  static-3dgs iter {it+1}/{iters}  "
                      f"loss={total.item():.5f}")

        # Conservative opacity pruning (opacity logit < -3.5 -> opacity < 0.029)
        with torch.no_grad():
            valid_mask = (cloud.opacities.squeeze(-1) > -3.5)
            if valid_mask.sum() > 500:  # Retain active visual hull cloud
                cloud = GaussianCloud(
                    means=cloud.means[valid_mask],
                    scales=cloud.scales[valid_mask],
                    rotations=cloud.rotations[valid_mask],
                    opacities=cloud.opacities[valid_mask],
                    sh=cloud.sh[valid_mask],
                )

        # Detach the final substrate; MCRAH does not train it.
        final = GaussianCloud(
            means=cloud.means.detach().clone(),
            scales=cloud.scales.detach().clone(),
            rotations=quaternion_normalize(cloud.rotations.detach().clone()),
            opacities=cloud.opacities.detach().clone(),
            sh=cloud.sh.detach().clone(),
        )
        return StaticInitResult(cloud=final, history=history, iterations=iters)


def save_cloud(cloud: GaussianCloud, path: Path | str) -> None:
    """Save a static cloud to ``path`` as .pt (torch state-dict of tensors)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "means": cloud.means.detach().cpu(),
        "scales": cloud.scales.detach().cpu(),
        "rotations": cloud.rotations.detach().cpu(),
        "opacities": cloud.opacities.detach().cpu(),
        "sh": cloud.sh.detach().cpu(),
    }, path)


def load_cloud(path: Path | str, device: Optional[str] = None) -> GaussianCloud:
    d = torch.load(Path(path), map_location=device or "cpu", weights_only=True)
    return GaussianCloud(
        means=d["means"], scales=d["scales"], rotations=d["rotations"],
        opacities=d["opacities"], sh=d["sh"],
    )
