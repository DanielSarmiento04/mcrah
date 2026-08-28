"""Phase 4: Evaluation, autoregressive rollout stability, and ablations.

Implements workflow.md Phase 4 Steps 10-12:
  * Step 10 - Visual benchmarking with PSNR / SSIM / LPIPS.
  * Step 11 - Autoregressive rollout testing: 100+ steps without GT, measuring
    error-drift distributions.
  * Step 12 - Ablation studies: systematically disable DHGC / noise injection /
    2-stage vs joint training and record the delta in NVS metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from ..data import SceneSample
from ..gs import render, set_rasterizer
from ..losses.rendering import psnr, ssim_metric
from ..models import MCRAH


@dataclass
class EvalMetrics:
    """Aggregated NVS metrics over an evaluation set."""
    psnr_mean: float = 0.0
    ssim_mean: float = 0.0
    lpips_mean: float = 0.0
    n_samples: int = 0
    per_sample: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class RolloutStability:
    """Error-drift statistics over a long autoregressive rollout (Step 11)."""
    steps: List[int] = field(default_factory=list)
    pos_drift: List[float] = field(default_factory=list)   # ||Δmean|| per step
    rot_drift: List[float] = field(default_factory=list)   # quaternion angle
    cloud_norms: List[float] = field(default_factory=list)


class Evaluator:
    """NVS benchmarking + rollout stability (Phase 4 Step 10/11)."""

    def __init__(self, cfg: Config, device: Optional[str] = None):
        self.cfg = cfg
        self.device = device or cfg.device_str()
        set_rasterizer("auto")
        self._lpips = None  # lazily loaded (optional dep)

    def _get_lpips(self):
        if self._lpips is None:
            try:
                import warnings
                import lpips  # type: ignore
                # lpips internally calls torchvision with the deprecated
                # ``pretrained=True`` kwarg; suppress the warning here.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    self._lpips = lpips.LPIPS(net="alex").to(self.device)
                    self._lpips.eval()
            except Exception:
                self._lpips = False  # mark unavailable
        return self._lpips if self._lpips is not False else None

    @torch.no_grad()
    def evaluate(
        self,
        model: MCRAH,
        samples: List[SceneSample],
        max_samples: Optional[int] = None,
    ) -> EvalMetrics:
        """Render the model's rollout at each sample's camera and score NVS."""
        model.eval()
        dev = self.device
        lpips_fn = self._get_lpips()
        metrics = EvalMetrics()
        psnr_sum = ssim_sum = lpips_sum = 0.0
        count = 0

        # Group samples by (category, time_idx) to build rollouts.
        by_time: Dict[Tuple[str, int], List[SceneSample]] = {}
        for s in samples:
            key = (s.category, s.time_idx)
            by_time.setdefault(key, []).append(s)
        times = sorted(by_time.keys(), key=lambda k: k[1])

        for key in times[:max_samples] if max_samples else times:
            window = by_time[key][:1]  # one view per time step for metrics
            t = torch.tensor(window[0].time, device=dev)
            step = model.step(model.cloud, t)
            s = window[0]
            # Render at the capped supervision resolution; resample the target
            # to match so PSNR/SSIM are computed at a consistent scale.
            H = self.cfg.data.render_wh[1]
            W = self.cfg.data.render_wh[0]
            bg = torch.ones(3, device=dev) if self.cfg.data.white_background else None
            out = render(step.cloud, s.c2w.to(dev), s.intrinsics.to(dev),
                        width=W, height=H, bg_color=bg)
            pred = out.image.unsqueeze(0).clamp(0, 1)  # (1,3,H,W)
            tgt = s.image.to(dev).unsqueeze(0)         # (1,3,H0,W0)
            if tgt.shape[-2:] != (H, W):
                tgt = torch.nn.functional.interpolate(
                    tgt, size=(H, W), mode="bilinear", align_corners=False)

            p = float(psnr(pred, tgt).item())
            ss = float(ssim_metric(pred, tgt).item())
            ll = 0.0
            if lpips_fn is not None:
                # LPIPS expects [-1, 1].
                ll = float(
                    lpips_fn(pred * 2 - 1, tgt * 2 - 1).mean().item())
            psnr_sum += p
            ssim_sum += ss
            lpips_sum += ll
            count += 1
            metrics.per_sample.append(
                {"psnr": p, "ssim": ss, "lpips": ll, "category": key[0],
                 "time_idx": key[1]})

        if count:
            metrics.psnr_mean = psnr_sum / count
            metrics.ssim_mean = ssim_sum / count
            metrics.lpips_mean = lpips_sum / count
            metrics.n_samples = count
        return metrics

    @torch.no_grad()
    def rollout_stability(
        self,
        model: MCRAH,
        n_steps: int = 100,
        dt: float = 0.05,
    ) -> RolloutStability:
        """Long-horizon rollout without ground truth (Step 11).

        Measures how the predicted cloud drifts from the static (t=0) reference
        over ``n_steps`` autoregressive steps — the key error-drift diagnostic.
        """
        model.eval()
        dev = self.device
        result = RolloutStability()
        base_means = model.cloud.means.detach().clone()
        base_rot = model.cloud.rotations.detach().clone()

        cloud = model.cloud
        for i in range(n_steps):
            t = torch.tensor(i * dt, device=dev)
            step = model.step(cloud, t)
            cloud = step.cloud
            dpos = (cloud.means - base_means).norm(dim=-1).mean().item()
            # Rotation drift: angle between current and base quaternions.
            dot = (cloud.rotations * base_rot).sum(-1).abs().clamp_max(1.0)
            ang = 2.0 * torch.acos(dot.clamp(-1.0, 1.0)).mean().item()
            result.steps.append(i)
            result.pos_drift.append(dpos)
            result.rot_drift.append(ang)
            result.cloud_norms.append(float(cloud.means.norm().item()))
        return result


@dataclass
class AblationResult:
    """One ablation arm (Phase 4 Step 12)."""
    name: str
    config: str   # description of what was disabled
    metrics: Optional[EvalMetrics] = None
    stability: Optional[RolloutStability] = None


def build_ablation_configs(cfg: Config) -> List[Tuple[str, Config]]:
    """Return the standard ablation arms (rules.md Rule 9).

    Each entry is (arm_name, modified_config). The caller trains + evaluates
    each independently to isolate the contribution of each component.
    """
    arms: List[Tuple[str, Config]] = []
    import copy

    # Full model (reference).
    arms.append(("full", cfg))

    # No DHGC: disable far-field attention (Stage 1 only).
    c = copy.deepcopy(cfg)
    c.model.num_dhgc_layers = 0
    arms.append(("no_dhgc", c))

    # No noise injection.
    c = copy.deepcopy(cfg)
    c.model.noise_std = 0.0
    arms.append(("no_noise", c))

    # No PDE regularization.
    c = copy.deepcopy(cfg)
    c.model.pde_weight = 0.0
    arms.append(("no_pde", c))

    # Joint-only (skip decoupled two-stage).
    c = copy.deepcopy(cfg)
    c.train.stage = "joint"
    arms.append(("joint_only", c))

    return arms
