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
from ..models.mcrah import RigidityLoss, TopologySmoothnessLoss


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

        self.state = TrainState(stage=cfg.train.stage)
        self.model.configure_stage(cfg.train.stage)

        self._opt = self._build_optimizer()

    # ------------------------------------------------------------------ #
    # Optimization
    # ------------------------------------------------------------------ #
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Adam with separate LRs for the two stages (decoupled training)."""
        cfg = self.cfg
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            # Avoid empty-param error in edge cases (e.g. eval-only).
            params = [torch.zeros(1, requires_grad=True, device=self.device)]
        return torch.optim.AdamW(
            params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    def set_stage(self, stage: str) -> None:
        """Switch the decoupled training stage and rebuild the optimizer."""
        if stage not in ("dense", "farfield", "joint"):
            raise ValueError(f"unknown stage: {stage}")
        self.model.configure_stage(stage)
        self.state.stage = stage
        self._opt = self._build_optimizer()
        print(f"[trainer] stage -> {stage} "
              f"({sum(p.numel() for p in self.model.parameters() if p.requires_grad)} "
              f"trainable params)")

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
            start = torch.randint(0, len(times) - tw + 1, (1,)).item()
            window = times[start:start + tw]
        # Pick one representative view per time step.
        return [by_time[t][0] for t in window]

    def _rollout_and_render(
        self,
        window: List[SceneSample],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List]:
        """Run the autoregressive rollout and render each predicted cloud.

        Returns: (pred_images, target_images, deltas, steps)

        Renders at ``cfg.data.render_wh`` to bound the pure-torch rasterizer's
        O(N*H*W) autograd memory (rules.md Rule 6). Targets are resampled to the
        same resolution so the photometric loss stays well-posed.
        """
        dev = self.device
        H = self.cfg.data.render_wh[1]
        W = self.cfg.data.render_wh[0]
        bg = torch.ones(3, device=dev) if self.cfg.data.white_background else None
        times = [torch.tensor(s.time, device=dev) for s in window]
        steps = self.model.rollout(times, noise_injector=self.noise)

        preds, tgts = [], []
        for s, step in zip(window, steps):
            K = s.intrinsics.to(dev)
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
        """Combined loss: L1+SSIM + relative-L2 + PDE smoothness + MCRAH.

        The MCRAH rigidity + topology-smoothness losses are only active when the
        model has the adaptive hypergraph (cfg.hypergraph.adaptive=True).
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
            pde = pred.new_zeros(())

        # MCRAH novel losses (only if adaptive hypergraph is active).
        rig, topo = pred.new_zeros(()), pred.new_zeros(())
        if (steps is not None and self.cfg.hypergraph.adaptive
                and self.model.mcrah is not None):
            mc = cfg.mcrah_loss
            memberships = [s.membership for s in steps if s.membership is not None]
            clouds_t = [s.cloud for s in steps if s.membership is not None]
            if memberships:
                ref0 = self.model.mcrah.means_0
                assign0 = self.model.mcrah.hypergraph.assignment_0
                # Rigidity: displacement should match cluster centroid displacement.
                rig = sum(
                    self.rigidity_loss(c.means, ref0, m, assign0)
                    for m, c in zip(memberships, clouds_t)) / len(memberships)
                # Topology smoothness: membership should not jump between steps.
                if len(memberships) >= 2:
                    topo = sum(
                        self.topo_loss(memberships[i], memberships[i - 1])
                        for i in range(1, len(memberships))) / (len(memberships) - 1)

        loss = (photo + cfg.train.w_rel_l2 * rel + pde
                + mc.rigidity_weight * rig
                + mc.topology_smoothness_weight * topo)
        metrics = {
            "loss": float(loss.item()),
            "photo": float(photo.item()),
            "rel_l2": float(rel.item()),
            "pde": float(pde.item()),
            "rigidity": float(rig.item()),
            "topology": float(topo.item()),
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
        pred, target, deltas, steps = self._rollout_and_render(window)
        loss, metrics = self._compute_loss(
            pred, target, deltas, self.model.cloud, steps)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.train.grad_clip)
        self._opt.step()
        self.noise.step()
        self.state.step += 1

        if metrics["loss"] < self.state.best_loss:
            self.state.best_loss = metrics["loss"]
        self.state.history.append(metrics)
        return metrics

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
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self._opt.state_dict(),
            "state": self.state.__dict__,
            "cfg": self.cfg.__dict__,
        }, path)
        return path

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(Path(path), map_location=self.device,
                          weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.state.__dict__.update(ckpt["state"])
        self.model.configure_stage(self.state.stage)
