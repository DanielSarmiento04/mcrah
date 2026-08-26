#!/usr/bin/env python
"""Phase 4 evaluation only: load a trained checkpoint and benchmark.

Usage:
    python scripts/evaluate.py --category trex --checkpoint runs/trex/runs/checkpoint_joint.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mcrah.config import Config
from mcrah.data import DNeRFDataset
from mcrah.training import Evaluator, load_cloud
from mcrah.models import MCRAH


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--category", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--static-cloud", default=None,
                   help="path to static_cloud.pt (default: checkpoint dir)")
    p.add_argument("--data-root", default="data")
    p.add_argument("--rollout-steps", type=int, default=100)
    p.add_argument("--max-eval", type=int, default=100)
    args = p.parse_args()

    cfg = Config()
    cfg.data.data_root = Path(args.data_root)
    device = cfg.device_str()

    ckpt_dir = Path(args.checkpoint).parent.parent
    cloud_path = args.static_cloud or str(ckpt_dir / "static_cloud.pt")
    cloud = load_cloud(cloud_path, device=device)

    model = MCRAH(cfg, cloud).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"loaded checkpoint: {args.checkpoint}")

    evaluator = Evaluator(cfg, device=device)
    ds = DNeRFDataset(cfg, split="test", categories=[args.category])
    samples = [ds[i] for i in range(min(len(ds), args.max_eval))]

    metrics = evaluator.evaluate(model, samples)
    print(f"\nNVS Metrics ({args.category}, n={metrics.n_samples}):")
    print(f"  PSNR  = {metrics.psnr_mean:.3f}")
    print(f"  SSIM  = {metrics.ssim_mean:.4f}")
    print(f"  LPIPS = {metrics.lpips_mean:.4f}")

    stability = evaluator.rollout_stability(
        model, n_steps=args.rollout_steps)
    print(f"\nRollout stability ({args.rollout_steps} steps):")
    print(f"  final pos drift = {stability.pos_drift[-1]:.5f}")
    print(f"  final rot drift = {stability.rot_drift[-1]:.5f} rad")
    print(f"  cloud norm      = {stability.cloud_norms[-1]:.3f}")

    out = {
        "category": args.category,
        "psnr": metrics.psnr_mean,
        "ssim": metrics.ssim_mean,
        "lpips": metrics.lpips_mean,
        "n_samples": metrics.n_samples,
        "rollout_final_pos_drift": stability.pos_drift[-1],
        "rollout_final_rot_drift": stability.rot_drift[-1],
        "per_sample": metrics.per_sample,
    }
    out_path = ckpt_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nresults written to {out_path}")


if __name__ == "__main__":
    main()
