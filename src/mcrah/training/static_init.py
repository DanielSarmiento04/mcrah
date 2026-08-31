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
        seeded from the first view's depth (heuristic uniform sampling).
        """
        cfg = self.cfg
        iters = iterations or cfg.static_gs.iterations
        dev = self.device

        # Move views to device. Render at the capped supervision resolution
        # (cfg.data.render_wh) so the pure-torch rasterizer's O(N*H*W) memory
        # stays bounded on Apple Silicon (rules.md Rule 6).  On CUDA with the
        # official rasterizer, render_wh may be 800x800 (set by auto_render_wh).
        rH, rW = cfg.data.render_wh[1], cfg.data.render_wh[0]
        views = [
            (F.interpolate(img.unsqueeze(0), size=(rH, rW), mode="bilinear",
                           align_corners=False).squeeze(0).to(dev),
             c2w.to(dev), K.to(dev))
            for img, c2w, K in views
        ]
        H, W = views[0][0].shape[-2], views[0][0].shape[-1]

        # Seed points: if none provided, initialize Gaussians in the 3D volume
        # centered at the world origin (0, 0, 0) where the D-NeRF object resides.
        if init_points is None:
            n = cfg.static_gs.num_gaussians
            init_points = torch.randn(n, 3, device=dev) * 0.45

        cloud = seed_cloud_from_points(init_points, dev)

        # Parameter groups with per-attribute learning rates (3DGS convention).
        params = [
            {"params": [cloud.means], "lr": cfg.static_gs.lr_means},
            {"params": [cloud.scales], "lr": cfg.static_gs.lr_scales},
            {"params": [cloud.opacities], "lr": cfg.static_gs.lr_opacity},
            {"params": [cloud.sh], "lr": cfg.static_gs.lr_sh},
        ]
        if cfg.model.predict_rotation:
            params.append(
                {"params": [cloud.rotations], "lr": cfg.static_gs.lr_means})
        opt = torch.optim.Adam(params, lr=cfg.static_gs.lr_means)
        sched = torch.optim.lr_scheduler.ExponentialLR(
            opt, gamma=0.99)

        history: List[float] = []
        # D-NeRF renders objects on a white background. The rasterizer defaults
        # to black, so without an explicit bg_color the loss is dominated by the
        # background mismatch and the cloud has no signal to learn the object.
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
            history.append(float(total.item()))
            if (it + 1) % max(1, iters // 10) == 0:
                print(f"  static-3dgs iter {it+1}/{iters}  "
                      f"loss={total.item():.5f}")

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
