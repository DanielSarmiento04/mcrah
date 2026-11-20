"""Phase 3 Steps 8-9: MCRAH training loop.

Implements the decoupled two-stage training (rules.md Rule 8):
  * Stage "dense"     - train SIMGNN only (DHGC frozen).
  * Stage "farfield"   - train DHGC only (SIMGNN frozen).
  * Stage "joint"      - fine-tune both end-to-end.

Each iteration:
  1. Sample a temporal window of D-NeRF frames (target views at t=0..T).
  2. Build the autoregressive rollout from the static cloud:
        cloud_0 -> step(t_0) -> ... -> step(t_{T-1})
  3. Render each predicted cloud at the target camera and compute the combined
     loss: L1+SSIM (photometric) + relative-L2 (drift) + PDE smoothness.
  4. Backprop with grad clipping and the noise injector advancing its schedule.

The loop is device-agnostic (CPU/MPS/CUDA) via the pure-torch rasterizer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..config import Config
from ..data import SceneSample, collate_scenes
from ..gs import render, set_rasterizer
from ..gs.gaussian import GaussianCloud
from ..losses import L1SSIMLoss, NoiseInjector
from ..losses.pde import PDERegularizer, RelativeL2Loss
from ..models import MCRAH
from ..models.hypergraph import (
    DilatedStructure, Hypergraph, build_hypergraph_from_features,
    load_dilated_structure, load_hypergraph,
)
from ..models.mcrah import OrthogonalityLoss, RigidityLoss, TopologySmoothnessLoss


@dataclass
class TrainState:
    """Mutable training bookkeeping."""
    step: int = 0
    stage: str = "dense"
    best_loss: float = float("inf")
    history: List[Dict[str, float]] = field(default_factory=list)


class MCRAHTrainer:
    """Orchestrates the decoupled training of a :class:`MCRAH` model.

    Args:
        cfg: global config.
        cloud: the static (t=0) Gaussian cloud (Phase 1 output).
        hypergraph: optional pre-built hypergraph; if None, built from the cloud.
        dilated: optional pre-built dilated structure (needed for farfield/joint
            stages); if None, DHGC is skipped (Stage-1-only training).
        out_dir: where checkpoints + logs are written.
    """

    def __init__(
        self,
        cfg: Config,
        cloud: GaussianCloud,
        hypergraph: Optional[Hypergraph] = None,
        dilated: Optional[DilatedStructure] = None,
        out_dir: Optional[str | Path] = None,
    ):
        self.cfg = cfg
        self.device = cfg.device_str()
        self.out_dir = Path(out_dir) if out_dir else Path("runs/default")
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if self.device == "cpu":
            set_rasterizer("torch")
        else:
            set_rasterizer("auto")

        # Build the model.
        self.model = MCRAH(cfg, cloud, hypergraph=hypergraph,
                            dilated=dilated)
        self.model = self.model.to(self.device)

        # Losses + regularizers.
        self.photo_loss = L1SSIMLoss(
            w_l1=cfg.train.w_l1, w_ssim=cfg.train.w_ssim)
        self.rel_l2 = RelativeL2Loss()
        self.pde_reg = PDERegularizer(lam=cfg.model.pde_weight)
        self.noise = NoiseInjector(
            std=cfg.model.noise_std,
            warmup_steps=cfg.model.noise_warmup_steps)
        # MCRAH novel-module losses.
        self.rigidity_loss = RigidityLoss()
        self.topo_loss = TopologySmoothnessLoss()
        self.orth_loss = OrthogonalityLoss()

        self._lpips_module = None
        self._scheduler = None
        # Automatic Mixed Precision (AMP) for GPU speedup.
        # On CPU/MPS this is a no-op (enabled=False).
        self._use_amp = bool(cfg.train.use_amp) and self.device == "cuda"
        self._scaler = None
        self.state = TrainState(stage=cfg.train.stage)
        self.set_stage(cfg.train.stage)

    # ------------------------------------------------------------------ #
    # Perceptual Loss
    # ------------------------------------------------------------------ #
    def _get_lpips(self):
        """Lazily initialize frozen LPIPS AlexNet evaluator/loss module."""
        if self._lpips_module is None:
            try:
                import warnings
                import lpips
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    net = lpips.LPIPS(net="alex").to(self.device)
                    net.eval()
                    for p in net.parameters():
                        p.requires_grad = False
                    self._lpips_module = net
            except Exception:
                self._lpips_module = False
        return self._lpips_module if self._lpips_module is not False else None

    # ------------------------------------------------------------------ #
    # Optimization & Stage Control
    # ------------------------------------------------------------------ #
    def _build_optimizer(self, stage: Optional[str] = None) -> torch.optim.Optimizer:
        """AdamW with stage-specific initial learning rates (decoupled training)."""
        cfg = self.cfg
        stg = stage or self.state.stage
        if stg == "dense":
            lr = getattr(cfg.train, "lr_dense", cfg.train.lr)
        elif stg == "farfield":
            lr = getattr(cfg.train, "lr_farfield", cfg.train.lr)
        elif stg == "joint":
            lr = getattr(cfg.train, "lr_joint", cfg.train.lr)
        else:
            lr = cfg.train.lr

        net_params = []
        cloud_pos_params = []
        cloud_scale_params = []
        cloud_sh_params = []
        cloud_op_params = []
        cloud_names = {
            "_cloud_means", "_cloud_rotations",
            "_cloud_scales", "_cloud_sh", "_cloud_sh_rest", "_cloud_opacities",
        }
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                if "_cloud_means" in name or "_cloud_rotations" in name:
                    cloud_pos_params.append(p)
                elif "_cloud_scales" in name:
                    cloud_scale_params.append(p)
                elif "_cloud_sh" in name:
                    cloud_sh_params.append(p)
                elif "_cloud_opacities" in name:
                    cloud_op_params.append(p)
                elif not any(cn in name for cn in cloud_names):
                    net_params.append(p)

        param_groups = []
        if net_params:
            param_groups.append({
                "params": net_params, "lr": lr, "weight_decay": cfg.train.weight_decay
            })
        if cloud_pos_params:
            # Allow multi-view cameras in joint stage to refine 3D canonical
            # positions without smearing dynamic structures.
            param_groups.append({
                "params": cloud_pos_params, "lr": min(lr * 0.35, 7.0e-5), "weight_decay": 0.0
            })
        if cloud_scale_params:
            param_groups.append({
                "params": cloud_scale_params, "lr": min(lr * 12.0, 2.5e-3), "weight_decay": 0.0
            })
        if cloud_sh_params:
            param_groups.append({
                "params": cloud_sh_params, "lr": min(lr * 10.0, 2.0e-3), "weight_decay": 0.0
            })
        if cloud_op_params:
            # High opacity LR (up to 1.5e-2) in joint stage so mid-air floaters
            # seen against white background from the 200 multi-view cameras are
            # rapidly driven to zero opacity.
            param_groups.append({
                "params": cloud_op_params, "lr": min(lr * 75.0, 1.5e-2), "weight_decay": 0.0
            })

        if not param_groups:
            param_groups.append({"params": [torch.zeros(1, requires_grad=True, device=self.device)], "lr": lr})

        return torch.optim.AdamW(param_groups)

    def set_stage(self, stage: str, total_steps: Optional[int] = None) -> None:
        """Switch the decoupled training stage and rebuild the optimizer and scheduler."""
        if stage not in ("dense", "farfield", "joint"):
            raise ValueError(f"unknown stage: {stage}")
        self.model.configure_stage(stage)
        self.state.stage = stage
        self._opt = self._build_optimizer(stage)
        if getattr(self.cfg.train, "use_cosine_scheduler", True):
            steps = total_steps or self.cfg.train.iterations
            self._stage_total_steps = max(1, int(steps))
            self._stage_step = 0
            lr_min = getattr(self.cfg.train, "lr_min", 1e-6)
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self._opt, T_max=max(1, steps), eta_min=lr_min
            )
        else:
            self._stage_total_steps = max(1, int(total_steps or self.cfg.train.iterations))
            self._stage_step = 0
            self._scheduler = None

        if self._use_amp:
            try:
                self._scaler = torch.amp.GradScaler("cuda", init_scale=4096.0)
            except (TypeError, AttributeError):
                self._scaler = torch.cuda.amp.GradScaler(init_scale=4096.0)

        lr_curr = self._opt.param_groups[0]["lr"]
        print(f"[trainer] stage -> {stage} "
              f"({sum(p.numel() for p in self.model.parameters() if p.requires_grad)} "
              f"trainable params, lr={lr_curr:.6f})")

    # ------------------------------------------------------------------ #
    # Rollout + rendering
    # ------------------------------------------------------------------ #
    def _sample_window(self, samples: List[SceneSample]
                       ) -> List[SceneSample]:
        """Sample a temporal window of up to ``time_window`` frames."""
        cfg = self.cfg
        tw = cfg.train.time_window
        # Group by category+time_idx so we get a monotone temporal sequence.
        by_time = {}
        for s in samples:
            key = (s.category, s.time_idx)
            by_time.setdefault(key, []).append(s)
        times = sorted(by_time.keys(), key=lambda k: k[1])
        if len(times) <= tw:
            window = times
        else:
            # Mix fine contiguous rollouts (stride=1) with multi-view strided
            # rollouts (stride in 2..8) so autoregressive steps span real camera
            # baseline changes and match evaluation step intervals.
            max_stride = max(1, min(8, (len(times) - 1) // max(1, tw - 1)))
            if max_stride > 1 and float(torch.rand(1).item()) < 0.55:
                stride = int(torch.randint(2, max_stride + 1, (1,)).item())
            else:
                stride = 1
            span = (tw - 1) * stride + 1
            max_start = max(1, len(times) - span + 1)
            # Bias 20% of windows to start near t=0 or end near t=1 so boundary
            # frames receive uniform supervision.
            r_mode = float(torch.rand(1).item())
            if r_mode < 0.10:
                start = 0
            elif r_mode < 0.20:
                start = max_start - 1
            else:
                start = int(torch.randint(0, max_start, (1,)).item())
            window = times[start:start + span:stride][:tw]
        # Pick a random representative view per time step for multi-view consistency.
        window_samples = []
        for t in window:
            cands = by_time[t]
            v_idx = torch.randint(0, len(cands), (1,)).item() if len(cands) > 1 else 0
            window_samples.append(cands[v_idx])
        return window_samples

    def _rollout_and_render(
        self,
        window: List[SceneSample],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List]:
        """Run the autoregressive rollout and render each predicted cloud.

        Returns: (pred_images, target_images, deltas, steps)

        Renders at hardware-adaptive resolution (auto_render_wh() on CUDA)
        to maximize visual fidelity while bounding pure-torch O(N*H*W) memory on CPU/MPS.
        """
        dev = self.device
        render_wh = self.cfg.auto_render_wh() if dev == "cuda" else self.cfg.data.render_wh
        H, W = render_wh[1], render_wh[0]
        bg = torch.ones(3, device=dev) if self.cfg.data.white_background else None
        times = [torch.tensor(s.time, device=dev) for s in window]
        steps = self.model.rollout(times, noise_injector=self.noise)

        preds, tgts = [], []
        for s, step in zip(window, steps):
            K = s.intrinsics.clone().to(dev)
            orig_w = float(K[0, 2].item() * 2.0 + 1.0)
            orig_h = float(K[1, 2].item() * 2.0 + 1.0)
            if abs(orig_w - W) > 2.0 and orig_w > 0 and orig_h > 0:
                K[0, 0] *= W / orig_w
                K[1, 1] *= H / orig_h
                K[0, 2] = 0.5 * (W - 1.0)
                K[1, 2] = 0.5 * (H - 1.0)
            c2w = s.c2w.to(dev)
            out = render(step.cloud, c2w, K, width=W, height=H, bg_color=bg)
            preds.append(out.image)
            # Resample the target to the render resolution.
            tgt = s.image.to(dev).unsqueeze(0)  # (1,3,H0,W0)
            if tgt.shape[-2:] != (H, W):
                tgt = torch.nn.functional.interpolate(
                    tgt, size=(H, W), mode="bilinear",
                    align_corners=False)
            tgts.append(tgt.squeeze(0))
        pred = torch.stack(preds, dim=0)  # (T,3,H,W)
        target = torch.stack(tgts, dim=0)  # (T,3,H,W)
        deltas = [s.delta_pos for s in steps]
        return pred, target, deltas, steps

    def _compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        deltas: List[torch.Tensor],
        cloud: GaussianCloud,
        steps: Optional[List] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Combined loss: L1+SSIM + relative-L2 + PDE smoothness + MCRAH + perceptual LPIPS.

        The MCRAH rigidity + topology-smoothness losses are only active when the
        model has the adaptive hypergraph (cfg.hypergraph.adaptive=True).
        Perceptual LPIPS loss is enabled during the 'joint' fine-tuning stage.
        """
        cfg = self.cfg
        # Photometric: pred/target are (T,3,H,W); the loss treats T as batch.
        photo = self.photo_loss(pred, target)

        # Relative L2 on predicted position offsets (drift mitigation).
        rel = 0.0
        for d in deltas:
            rel = rel + self.rel_l2(d, cloud.means.detach())
        rel = rel / max(1, len(deltas))

        # PDE temporal smoothness on the stacked offsets.
        if len(deltas) >= 3:
            stacked = torch.stack(deltas, dim=0)  # (T,N,3)
            pde = self.pde_reg(stacked)
        else:
            pde = pred.new_zeros((), dtype=torch.float32)

        # MCRAH novel losses (only if adaptive hypergraph is active).
        rig = pred.new_zeros((), dtype=torch.float32)
        topo = pred.new_zeros((), dtype=torch.float32)
        orth = pred.new_zeros((), dtype=torch.float32)
        jepa = pred.new_zeros((), dtype=torch.float32)
        mc = cfg.mcrah_loss
        if (steps is not None and self.cfg.hypergraph.adaptive
                and self.model.mcrah is not None):
            mc_steps = [s for s in steps if s.membership is not None]
            if mc_steps:
                ref0 = self.model.mcrah.means_0
                assign0 = self.model.mcrah.hypergraph.assignment_0
                # SO(3)-invariant SE(3) rigidity: penalize only non-rigid residual
                rig = sum(
                    self.rigidity_loss(
                        s.cloud.means, ref0, s.membership, assign0,
                        expected_disp=getattr(s, "delta_pos_se3", None),
                    )
                    for s in mc_steps
                ) / len(mc_steps)
                # Hyperedge subspace orthogonality (M_norm^T M_norm ~ I_E)
                orth = sum(self.orth_loss(s.membership) for s in mc_steps) / len(mc_steps)
                # Topology smoothness & Hypergraph-JEPA latent dynamics
                if len(mc_steps) >= 2:
                    topo = sum(
                        self.topo_loss(mc_steps[i].membership, mc_steps[i - 1].membership)
                        for i in range(1, len(mc_steps))
                    ) / (len(mc_steps) - 1)
                    jepa_terms = []
                    for i in range(len(mc_steps) - 1):
                        z_curr = getattr(mc_steps[i], "edge_latents", None)
                        z_next = getattr(mc_steps[i + 1], "edge_latents", None)
                        t_curr = getattr(mc_steps[i], "time_val", None)
                        t_next = getattr(mc_steps[i + 1], "time_val", None)
                        if z_curr is not None and z_next is not None and t_curr is not None and t_next is not None:
                            dt = (t_next - t_curr).abs()
                            jepa_terms.append(self.model.mcrah.jepa.jepa_vicreg_loss(z_curr, z_next, dt))
                    if jepa_terms:
                        jepa = sum(jepa_terms) / len(jepa_terms)

        # Perceptual LPIPS loss + multi-view floater sparsity in farfield/joint stages
        lp_loss = pred.new_zeros((), dtype=torch.float32)
        joint_op_reg = pred.new_zeros((), dtype=torch.float32)
        w_lpips = getattr(cfg.train, "w_lpips", 0.0)
        if self.state.stage in ("farfield", "joint"):
            # Mild opacity sparsity drives mid-air floaters (seen against white
            # background across the 200 multi-view cameras) to zero opacity.
            joint_op_reg = 1.5e-5 * torch.sigmoid(self.model._cloud_opacities).mean()
        if self.state.stage == "joint" and w_lpips > 0.0:
            lpips_fn = self._get_lpips()
            if lpips_fn is not None:
                p_norm = (pred.float() * 2.0 - 1.0).clamp(-1.0, 1.0)
                t_norm = (target.float() * 2.0 - 1.0).clamp(-1.0, 1.0)
                lp_loss = lpips_fn(p_norm, t_norm).mean()

        loss = (photo + cfg.train.w_rel_l2 * rel + pde
                + mc.rigidity_weight * rig
                + mc.topology_smoothness_weight * topo
                + getattr(mc, "orthogonality_weight", 0.0) * orth
                + getattr(mc, "jepa_weight", 0.0) * jepa
                + w_lpips * lp_loss
                + joint_op_reg)
        metrics = {
            "loss": float(loss.item()),
            "photo": float(photo.item()),
            "lpips": float(lp_loss.item()),
            "rel_l2": float(rel.item()),
            "pde": float(pde.item()),
            "rigidity": float(rig.item()),
            "topology": float(topo.item()),
            "orth": float(orth.item()),
            "jepa": float(jepa.item()),
        }
        return loss, metrics

    # ------------------------------------------------------------------ #
    # Train / eval steps
    # ------------------------------------------------------------------ #
    def train_step(self, samples: List[SceneSample]) -> Dict[str, float]:
        self.model.train()
        self.noise.train()
        window = self._sample_window(samples)
        if len(window) < 2:
            return {"loss": 0.0}

        self._opt.zero_grad()

        if self._use_amp:
            try:
                amp_ctx = torch.amp.autocast("cuda")
            except (TypeError, AttributeError):
                amp_ctx = torch.cuda.amp.autocast()
            with amp_ctx:
                pred, target, deltas, steps = self._rollout_and_render(window)
            # Compute all losses strictly in float32 outside autocast
            loss, metrics = self._compute_loss(
                pred, target, deltas, self.model.cloud, steps)
            if loss.requires_grad and loss.grad_fn is not None and torch.isfinite(loss):
                self._scaler.scale(loss).backward()
                self._scaler.unscale_(self._opt)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.train.grad_clip)
                scale_before = self._scaler.get_scale()
                self._scaler.step(self._opt)
                self._scaler.update()
                scale_after = self._scaler.get_scale()
                # In AMP, step succeeded if scale did not decrease (no infs/NaNs)
                step_successful = (scale_after >= scale_before)
            else:
                step_successful = False
        else:
            pred, target, deltas, steps = self._rollout_and_render(window)
            loss, metrics = self._compute_loss(
                pred, target, deltas, self.model.cloud, steps)
            if loss.requires_grad and loss.grad_fn is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.train.grad_clip)
                self._opt.step()
                step_successful = True
            else:
                step_successful = False

        self.noise.step()
        self._stage_step = getattr(self, "_stage_step", 0) + 1
        if step_successful and self.state.stage in ("farfield", "joint"):
            with torch.no_grad():
                self.model._cloud_scales.data.clamp_(-7.5, -3.10)
                self.model._cloud_sh.data.clamp_(0.0, 1.0)
                if hasattr(self.model, "_cloud_sh_rest"):
                    self.model._cloud_sh_rest.data.clamp_(-1.5, 1.5)
                # Suppress transparent mid-air floaters (logit < -3.0 -> -8.0)
                op_data = self.model._cloud_opacities.data
                op_data.clamp_(-8.0, 5.0)
                op_data[op_data < -3.0] = -8.0
                q = self.model._cloud_rotations.data
                self.model._cloud_rotations.data.copy_(
                    q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                )
                # Dynamic Gaussian Recycling at 25% and 55% of joint stage:
                # reinvest dead floater Gaussians directly into active moving
                # foreground structures (e.g. lego scoop, bouncing balls, limbs).
                stg_tot = getattr(self, "_stage_total_steps", 0)
                if (
                    self.state.stage == "joint"
                    and stg_tot >= 200
                    and self._stage_step in (int(stg_tot * 0.25), int(stg_tot * 0.55))
                    and self.model.cloud.n >= 500
                ):
                    self._recycle_dead_gaussians(deltas)
                if self.model.mcrah is not None:
                    self.model.mcrah.means_0.copy_(self.model._cloud_means.detach())
        if self._scheduler is not None and step_successful:
            self._scheduler.step()
        metrics["lr"] = float(self._opt.param_groups[0]["lr"])
        self.state.step += 1

        if metrics["loss"] < self.state.best_loss:
            self.state.best_loss = metrics["loss"]
        self.state.history.append(metrics)
        return metrics

    @torch.no_grad()
    def _recycle_dead_gaussians(self, deltas: List[torch.Tensor]) -> None:
        """Clone transparent/dead Gaussians (opacities <= -2.8) onto active
        foreground Gaussians with the largest dynamic motion/gradient."""
        op_flat = self.model._cloud_opacities.data.squeeze(-1)
        dead_mask = op_flat <= -2.8
        active_mask = op_flat > 0.2
        n_dead = int(dead_mask.sum().item())
        n_active = int(active_mask.sum().item())
        if n_dead < 4 or n_active < 16:
            return

        dead_idx = torch.where(dead_mask)[0]
        active_idx = torch.where(active_mask)[0]
        max_recycle = min(n_dead, max(64, self.model.cloud.n // 10))
        dead_idx = dead_idx[:max_recycle]

        # Score active Gaussians by dynamic motion magnitude + scale + grad norm
        if deltas:
            disp_mag = torch.stack([d.detach().norm(dim=-1) for d in deltas], dim=0).mean(dim=0)
        else:
            disp_mag = torch.zeros_like(op_flat)
        grad_mag = (
            self.model._cloud_means.grad.detach().norm(dim=-1)
            if self.model._cloud_means.grad is not None
            else torch.zeros_like(op_flat)
        )
        scores = (0.1 + disp_mag[active_idx] * 5.0 + grad_mag[active_idx] * 20.0).clamp_min(1e-4)
        src_local = torch.multinomial(scores, dead_idx.numel(), replacement=True)
        src_idx = active_idx[src_local]

        self.model._cloud_means.data[dead_idx] = (
            self.model._cloud_means.data[src_idx]
            + torch.randn_like(self.model._cloud_means.data[src_idx]) * 0.0025
        )
        self.model._cloud_sh.data[dead_idx] = self.model._cloud_sh.data[src_idx].clone()
        if hasattr(self.model, "_cloud_sh_rest"):
            self.model._cloud_sh_rest.data[dead_idx] = self.model._cloud_sh_rest.data[src_idx].clone()
        self.model._cloud_rotations.data[dead_idx] = self.model._cloud_rotations.data[src_idx].clone()
        self.model._cloud_scales.data[dead_idx] = (
            self.model._cloud_scales.data[src_idx] - 0.20
        ).clamp_(-7.0, -4.30)
        self.model._cloud_opacities.data[dead_idx] = 0.85

        # Also copy cluster assignment so recycled Gaussians inherit their parent's hyperedge
        self.model.cluster_id[dead_idx] = self.model.cluster_id[src_idx]
        if self.model.mcrah is not None:
            from ..models.mcrah import compute_centroids
            self.model.mcrah.hypergraph.assignment_0[dead_idx] = (
                self.model.mcrah.hypergraph.assignment_0[src_idx]
            )
            self.model.mcrah.means_0.copy_(self.model._cloud_means.detach())
            new_c0 = compute_centroids(
                self.model.mcrah.means_0,
                self.model.mcrah.hypergraph.assignment_0,
                self.model.mcrah.hypergraph.n_edges,
            )
            self.model.mcrah.centroid_0.copy_(new_c0)

        # Reset Adam momentum for recycled canonical cloud indices
        cloud_tensors = [
            self.model._cloud_means,
            self.model._cloud_scales,
            self.model._cloud_rotations,
            self.model._cloud_sh,
            self.model._cloud_opacities,
        ]
        if hasattr(self.model, "_cloud_sh_rest"):
            cloud_tensors.append(self.model._cloud_sh_rest)
        for p in cloud_tensors:
            st = self._opt.state.get(p)
            if st is not None and p.ndim >= 1 and p.shape[0] == self.model.cloud.n:
                if "exp_avg" in st and st["exp_avg"].ndim >= 1 and st["exp_avg"].shape[0] == self.model.cloud.n:
                    st["exp_avg"][dead_idx] = 0.0
                if "exp_avg_sq" in st and st["exp_avg_sq"].ndim >= 1 and st["exp_avg_sq"].shape[0] == self.model.cloud.n:
                    st["exp_avg_sq"][dead_idx] = 0.0

    @torch.no_grad()
    def eval_step(self, samples: List[SceneSample]) -> Dict[str, float]:
        self.model.eval()
        window = self._sample_window(samples)
        if len(window) < 2:
            return {"loss": 0.0}
        pred, target, deltas, steps = self._rollout_and_render(window)
        loss, metrics = self._compute_loss(
            pred, target, deltas, self.model.cloud, steps)
        return metrics

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #
    def save(self, tag: str = "latest") -> Path:
        path = self.out_dir / f"checkpoint_{tag}.pt"
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self._opt.state_dict(),
            "state": self.state.__dict__,
            "cfg": self.cfg.__dict__,
        }
        if self._scheduler is not None:
            payload["scheduler"] = self._scheduler.state_dict()
        torch.save(payload, path)
        return path

    def load(self, path: str | Path, strict: bool = False) -> None:
        ckpt = torch.load(Path(path), map_location=self.device,
                          weights_only=False)
        self.model.load_state_dict(ckpt["model"], strict=strict)
        self.state.__dict__.update(ckpt["state"])
        self.set_stage(self.state.stage)
        if "optimizer" in ckpt:
            try:
                self._opt.load_state_dict(ckpt["optimizer"])
            except Exception:
                pass
        if "scheduler" in ckpt and self._scheduler is not None:
            try:
                self._scheduler.load_state_dict(ckpt["scheduler"])
            except Exception:
                pass
