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

import math
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


def _project_and_score_candidates(
    candidates: torch.Tensor,
    views: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project 3D candidates (M, 3) into all views and count foreground hits vs background misses."""
    device = candidates.device
    n_cand = candidates.shape[0]
    accum_color = torch.zeros(n_cand, 3, device=device)
    fg_hits = torch.zeros(n_cand, device=device)
    bg_misses = torch.zeros(n_cand, device=device)

    for img, c2w, K in views:
        H, W = img.shape[-2], img.shape[-1]
        w2c = torch.inverse(c2w)
        R_w2c = w2c[:3, :3]
        t_w2c = w2c[:3, 3]
        pts_cam = candidates @ R_w2c.T + t_w2c

        z = pts_cam[:, 2]
        in_front = z > 0.1

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        u = (fx * pts_cam[:, 0] / z.clamp_min(1e-4) + cx).round().long()
        v = (fy * pts_cam[:, 1] / z.clamp_min(1e-4) + cy).round().long()

        in_bounds = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not in_bounds.any():
            bg_misses += 1.0
            continue

        # Any point projecting outside the image frustum of a camera when the
        # object is centered inside the frame is outside the visual hull.
        out_idx = torch.where(~in_bounds)[0]
        if out_idx.numel() > 0:
            bg_misses.index_add_(
                0, out_idx, torch.ones_like(out_idx, dtype=torch.float)
            )

        valid_idx = torch.where(in_bounds)[0]
        u_v = u[valid_idx]
        v_v = v[valid_idx]

        pix_colors = img[:, v_v, u_v].t()  # (M_valid, 3)
        # Use 0.985 threshold so bright white/ivory bone (skull, ribs, tail tip)
        # against the 1.0 white background is counted as foreground, not background.
        is_fg = (pix_colors < 0.985).any(dim=-1)

        fg_valid_idx = valid_idx[is_fg]
        bg_valid_idx = valid_idx[~is_fg]

        if fg_valid_idx.numel() > 0:
            accum_color.index_add_(0, fg_valid_idx, pix_colors[is_fg])
            fg_hits.index_add_(
                0, fg_valid_idx, torch.ones_like(fg_valid_idx, dtype=torch.float)
            )
        if bg_valid_idx.numel() > 0:
            bg_misses.index_add_(
                0, bg_valid_idx, torch.ones_like(bg_valid_idx, dtype=torch.float)
            )

    return accum_color, fg_hits, bg_misses


def _carve_rays_from_pixels(
    u: torch.Tensor,
    v: torch.Tensor,
    c2w: torch.Tensor,
    K: torch.Tensor,
    views: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    n_depths: int = 48,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """March 3D candidates along camera rays through (u, v) and pick the depth
    that maximizes multi-view silhouette consistency across all ``views``.

    Returns (best_pts (R,3), best_colors (R,3), best_scores (R,)).
    """
    device = c2w.device
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    dirs_cam = torch.stack(
        [(u.float() - cx) / fx, (v.float() - cy) / fy, torch.ones_like(u, dtype=torch.float)],
        dim=-1,
    )
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    dirs_w = dirs_cam @ R.T  # (R_rays, 3)
    cam_dist = float(t.norm().item())
    if cam_dist < 0.5:
        cam_dist = 4.0
    z_min = max(0.2, cam_dist - 1.30)
    z_max = cam_dist + 1.30
    z_vals = torch.linspace(z_min, z_max, n_depths, device=device)

    # Candidate 3D points along each ray: (R_rays, n_depths, 3)
    pts_ray = t.view(1, 1, 3) + dirs_w.unsqueeze(1) * z_vals.view(1, n_depths, 1)
    flat_pts = pts_ray.reshape(-1, 3)
    accum_c, fg_c, bg_c = _project_and_score_candidates(flat_pts, views)

    scores = (fg_c - 0.55 * bg_c).reshape(-1, n_depths)
    best_d = scores.argmax(dim=1)
    ray_idx = torch.arange(scores.shape[0], device=device)
    best_scores = scores[ray_idx, best_d]
    best_pts = pts_ray[ray_idx, best_d]

    flat_idx = ray_idx * n_depths + best_d
    best_colors = accum_c[flat_idx] / fg_c[flat_idx].unsqueeze(-1).clamp_min(1.0)
    return best_pts, best_colors, best_scores


def seed_cloud_from_views(
    views: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    n_gaussians: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Seed 3D points via Multi-View Foreground Ray-Carving + Edge/Bone Importance Sampling.

    Unlike uniform 3D cube sampling (which places >95% of points inside the thick
    solid pedestal and starves thin 1D/2D structures like the T-Rex tail tip,
    ribs, and skull), ray-carving shoots rays directly through every 2D foreground
    pixel (weighted toward high-contrast bone edges and bright bone details) and
    places each 3D Gaussian at the exact depth along the ray that maximizes
    multi-view silhouette consistency.

    Returns (points (N,3), colors (N,3), initial_opacities (N,1)).
    """
    n_views = max(1, len(views))
    rays_per_view = max(64, (n_gaussians * 2) // n_views)

    ray_pts_list: List[torch.Tensor] = []
    ray_col_list: List[torch.Tensor] = []
    ray_scr_list: List[torch.Tensor] = []

    for img, c2w, K in views:
        H, W = img.shape[-2], img.shape[-1]
        fg_mask = (img < 0.985).any(dim=0)
        ys, xs = torch.where(fg_mask)
        if ys.numel() < 16:
            continue

        # Compute 2D spatial gradient + bone brightness weight so thin ribs,
        # tail vertebrae, and skull edges receive 3x higher ray density than
        # flat interior pedestal regions.
        grad_x = torch.zeros(H, W, device=device)
        grad_y = torch.zeros(H, W, device=device)
        grad_x[:, :-1] = (img[:, :, 1:] - img[:, :, :-1]).abs().mean(dim=0)
        grad_y[:-1, :] = (img[:, 1:, :] - img[:, :-1, :]).abs().mean(dim=0)
        edge_mag = (grad_x + grad_y)[ys, xs]
        pix_lum = img[:, ys, xs].mean(dim=0)
        # Bright bone pixels (lum > 0.55) and high-edge pixels get higher weight
        weights = 1.0 + 6.0 * edge_mag + 1.5 * (pix_lum > 0.55).float()

        n_sample = min(rays_per_view, ys.numel())
        sampled_idx = torch.multinomial(weights, n_sample, replacement=(ys.numel() < n_sample))
        u_s = xs[sampled_idx]
        v_s = ys[sampled_idx]

        b_pts, b_cols, b_scr = _carve_rays_from_pixels(u_s, v_s, c2w, K, views, n_depths=48)
        # Blend view pixel color with multi-view average for crisp initialization
        view_cols = img[:, v_s, u_s].t()
        b_cols = 0.6 * view_cols + 0.4 * b_cols
        ray_pts_list.append(b_pts)
        ray_col_list.append(b_cols)
        ray_scr_list.append(b_scr)

    if ray_pts_list:
        all_pts = torch.cat(ray_pts_list, dim=0)
        all_cols = torch.cat(ray_col_list, dim=0)
        all_scr = torch.cat(ray_scr_list, dim=0)

        min_valid_score = max(0.5, n_views * 0.20) if n_views >= 4 else -10.0
        valid_mask = all_scr >= min_valid_score
        if valid_mask.sum() < max(32, n_gaussians // 4):
            valid_mask = torch.ones_like(all_scr, dtype=torch.bool)

        valid_pts = all_pts[valid_mask]
        valid_cols = all_cols[valid_mask]
        valid_scr = all_scr[valid_mask]

        # Keep a balanced mix across all views/regions rather than only top-score
        # pedestal points: filter to valid multi-view hull, then sample with
        # mild preference for higher consistency.
        if valid_pts.shape[0] >= n_gaussians:
            probs = (valid_scr - valid_scr.min() + 1.0).clamp_min(0.1)
            sel_idx = torch.multinomial(probs, n_gaussians, replacement=False)
        else:
            repeat_cnt = (n_gaussians // valid_pts.shape[0]) + 1
            sel_idx = torch.arange(valid_pts.shape[0], device=device).repeat(repeat_cnt)[:n_gaussians]

        selected_pts = valid_pts[sel_idx] + torch.randn(n_gaussians, 3, device=device) * 0.0015
        selected_colors = valid_cols[sel_idx].clamp(0.0, 1.0)
        selected_opacities = torch.full((n_gaussians, 1), 0.8, device=device)
        return selected_pts, selected_colors, selected_opacities

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

        # Move views to device using hardware-adaptive resolution.
        render_wh = cfg.auto_render_wh() if dev == "cuda" else cfg.data.render_wh
        rH, rW = render_wh[1], render_wh[0]
        scaled_views = []
        for img, c2w, K in views:
            img_d = F.interpolate(
                img.unsqueeze(0), size=(rH, rW), mode="bilinear",
                align_corners=False,
            ).squeeze(0).to(dev)
            K_d = K.clone().to(dev)
            # Ensure intrinsics principal point (cx, cy) and focal length match (rW, rH)
            orig_w = float(K_d[0, 2].item() * 2.0 + 1.0)
            orig_h = float(K_d[1, 2].item() * 2.0 + 1.0)
            if abs(orig_w - rW) > 2.0 and orig_w > 0 and orig_h > 0:
                sx = rW / orig_w
                sy = rH / orig_h
                K_d[0, 0] *= sx
                K_d[1, 1] *= sy
                K_d[0, 2] = 0.5 * (rW - 1.0)
                K_d[1, 2] = 0.5 * (rH - 1.0)
            scaled_views.append((img_d, c2w.to(dev), K_d))
        views = scaled_views
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
            # Start at -4.85 (exp(-4.85) ~ 0.0078, ~3px radius) well below the
            # -4.20 (0.015) ceiling so splats resolve fine ribs and vertebrae.
            cloud.scales.data.clamp_(-6.0, -4.75)

        # Parameter groups with per-attribute learning rates (3DGS convention,
        # scaled by D-NeRF camera radius ~3.5x).
        lr_means_init = cfg.static_gs.lr_means * 3.5
        params = [
            {"params": [cloud.means], "lr": lr_means_init},
            {"params": [cloud.scales], "lr": cfg.static_gs.lr_scales},
            {"params": [cloud.opacities], "lr": cfg.static_gs.lr_opacity},
            {"params": [cloud.sh], "lr": 4.0e-3},
        ]
        if cfg.model.predict_rotation:
            params.append(
                {"params": [cloud.rotations], "lr": lr_means_init})
        opt = torch.optim.Adam(params, lr=cfg.static_gs.lr_means)

        # Decoupled 3DGS schedules:
        # means & rotations: exponential decay from lr_init to lr_final (0.02x)
        # scales, opacities, sh: sustained cosine decay down to 20% of initial lr
        gamma_means = (0.02) ** (1.0 / max(1, iters))
        sched_funcs = [
            lambda step: gamma_means ** step,                            # means
            lambda step: 0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * step / max(1, iters))),  # scales
            lambda step: 0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * step / max(1, iters))),  # opacities
            lambda step: 0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * step / max(1, iters))),  # sh
        ]
        if cfg.model.predict_rotation:
            sched_funcs.append(lambda step: gamma_means ** step)         # rotations
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=sched_funcs)

        history: List[float] = []
        bg = torch.ones(3, device=dev) if cfg.data.white_background else None
        n_views = max(1, len(views))
        recycle_steps = {int(iters * 0.30), int(iters * 0.60)} if iters >= 200 else set()

        for it in range(iters):
            opt.zero_grad()
            total = 0.0
            for img, c2w, K in views:
                out = render(cloud, c2w, K, width=W, height=H, bg_color=bg)
                pred = out.image.unsqueeze(0)  # (1,3,H,W)
                tgt = img.unsqueeze(0)
                photo_loss = self.loss_fn(pred, tgt)
                # Gentle opacity sparsity regularization during the first 70% of
                # static init to clear floaters, then 0 so solid surfaces (stone
                # pedestal and white bone) reach full opacity without background bleed.
                if it < int(iters * 0.70):
                    op_reg = 2.5e-5 * torch.sigmoid(cloud.opacities).mean()
                else:
                    op_reg = 0.0
                loss = photo_loss + op_reg
                (loss / n_views).backward()
                total = total + float(photo_loss.item())
            total = total / n_views
            opt.step()
            sched.step()

            # Keep parameters physically bounded during optimization (strictly
            # below -4.22 so scales never enter the 0.015 clamp ceiling).
            with torch.no_grad():
                cloud.scales.data.clamp_(-7.5, -4.22)
                cloud.sh.data.clamp_(0.0, 1.0)
                cloud.opacities.data.clamp_(-8.0, 5.0)
                if (it + 1) in recycle_steps and cloud.n >= 500:
                    self._recycle_transparent_to_high_error(cloud, views, W, H, bg)

            history.append(float(total))
            if (it + 1) % max(1, iters // 10) == 0:
                print(f"  static-3dgs iter {it+1}/{iters}  "
                      f"loss={total:.5f}")

        # Prune only genuinely invisible floaters (logit <= -3.5 -> opacity <= 0.029)
        with torch.no_grad():
            valid_mask = (cloud.opacities.squeeze(-1) > -3.5)
            if valid_mask.sum() > 500:  # Retain active visual hull cloud
                cloud = GaussianCloud(
                    means=cloud.means[valid_mask],
                    scales=cloud.scales[valid_mask],
                    rotations=cloud.rotations[valid_mask],
                    opacities=cloud.opacities[valid_mask],
                    sh=cloud.sh.data[valid_mask],
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

    @torch.no_grad()
    def _recycle_transparent_to_high_error(
        self,
        cloud: GaussianCloud,
        views: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        W: int,
        H: int,
        bg: Optional[torch.Tensor],
    ) -> None:
        """Teleport transparent Gaussians (logit < -1.8) directly onto the
        highest-photometric-error foreground pixels via multi-view ray-carving."""
        dead_idx = torch.where(cloud.opacities.squeeze(-1) < -1.8)[0]
        if dead_idx.numel() < 16:
            return
        n_recycle = min(int(dead_idx.numel()), max(64, cloud.n // 4))
        dead_idx = dead_idx[:n_recycle]
        per_view = max(8, (n_recycle // max(1, len(views))) + 4)

        new_pts_list: List[torch.Tensor] = []
        new_col_list: List[torch.Tensor] = []
        for img, c2w, K in views:
            out = render(cloud, c2w, K, width=W, height=H, bg_color=bg)
            err = (out.image - img).abs().mean(dim=0)  # (H, W)
            fg_mask = (img < 0.985).any(dim=0)
            ys, xs = torch.where(fg_mask & (err > 0.02))
            if ys.numel() < 8:
                continue
            weights = (err[ys, xs] ** 2).clamp_min(1e-6)
            k = min(per_view, ys.numel())
            sel = torch.multinomial(weights, k, replacement=False)
            u_s, v_s = xs[sel], ys[sel]
            b_pts, b_cols, _ = _carve_rays_from_pixels(u_s, v_s, c2w, K, views, n_depths=48)
            view_cols = img[:, v_s, u_s].t()
            new_pts_list.append(b_pts)
            new_col_list.append(0.7 * view_cols + 0.3 * b_cols)

        if not new_pts_list:
            return
        new_pts = torch.cat(new_pts_list, dim=0)
        new_cols = torch.cat(new_col_list, dim=0)
        count = min(dead_idx.numel(), new_pts.shape[0])
        target_idx = dead_idx[:count]
        cloud.means.data[target_idx] = new_pts[:count] + torch.randn_like(new_pts[:count]) * 0.001
        cloud.sh.data[target_idx] = new_cols[:count].clamp(0.0, 1.0)
        cloud.opacities.data[target_idx] = 0.6
        cloud.scales.data[target_idx] = -5.15


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
