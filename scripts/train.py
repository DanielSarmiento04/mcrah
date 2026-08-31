#!/usr/bin/env python
"""End-to-end MCRAH training pipeline (workflow.md Phases 1-4).

Stages:
  1. Static 3DGS init at t=0   (Phase 1 Step 3)
  2. Stage "dense"   training  (Phase 3 Step 9, SIMGNN only)
  3. Stage "farfield" training  (Phase 3 Step 9, DHGC only)
  4. Stage "joint"   fine-tune (Phase 3 Step 9)
  5. Evaluation + rollout stability (Phase 4 Steps 10-11)

Usage:
    python scripts/train.py --category trex --iterations 5000
    python scripts/train.py --all --quick
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mcrah.config import Config
from mcrah.data import DNeRFDataset, collate_scenes
from mcrah.training import (
    MCRAHTrainer, Evaluator, StaticGSInit,
    load_cloud, save_cloud,
)


def load_t0_views(cfg: Config, category: str, device: str, max_t: float = 0.15, min_views: int = 12):
    """Load early / near t=0 training views for static 3DGS initialization.
    
    A single 2D view is mathematically ill-posed for 3D reconstruction and causes
    3DGS to collapse into a flat 2D billboard. We gather multi-view cameras from
    the earliest time frames (t <= max_t, minimum min_views) to reconstruct the
    true 360-degree base geometry.
    """
    ds = DNeRFDataset(cfg, split="train", categories=[category])
    sorted_idxs = sorted(range(len(ds)), key=lambda i: ds[i].time)
    selected = [i for i in sorted_idxs if ds[i].time <= max_t]
    if len(selected) < min_views:
        selected = sorted_idxs[:min(len(sorted_idxs), min_views)]

    views = []
    for i in selected:
        s = ds[i]
        views.append((s.image, s.c2w, s.intrinsics))
    times = [ds[i].time for i in selected]
    print(f"[{category}] static init: {len(views)} multi-view cameras (t in [{min(times):.3f}, {max(times):.3f}])")
    return views


def run_category(cfg: Config, category: str, args) -> dict:
    device = cfg.device_str()
    print(f"\n{'='*60}\n[{category}] device={device}\n{'='*60}")

    out_dir = Path(args.out) / category
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: static 3DGS init ----------------------------------- #
    cloud_path = out_dir / "static_cloud.pt"
    if args.skip_static and cloud_path.exists():
        print(f"[{category}] loading existing static cloud")
        cloud = load_cloud(cloud_path, device=device)
    else:
        views = load_t0_views(cfg, category, device)
        static_init = StaticGSInit(cfg, device=device)
        result = static_init.fit(
            views,
            iterations=args.static_iters,
        )
        cloud = result.cloud
        save_cloud(cloud, cloud_path)
        print(f"[{category}] static cloud: {cloud.n} Gaussians, "
              f"final loss={result.history[-1]:.5f}")

    # ---- Build dataset for training ---------------------------------- #
    ds = DNeRFDataset(cfg, split="train", categories=[category])
    samples = [ds[i] for i in range(min(len(ds), args.max_samples))]

    # ---- Phase 3: decoupled two-stage training ----------------------- #
    trainer = MCRAHTrainer(cfg, cloud, out_dir=out_dir / "runs")

    stages = ["dense", "farfield", "joint"] if not args.stage else [args.stage]
    iters = args.iterations
    for stage in stages:
        print(f"\n[{category}] training stage: {stage} ({iters} iters)")
        trainer.set_stage(stage)
        for it in range(iters):
            # Sample a random mini-batch of frames.
            idx = torch.randint(0, len(samples), (cfg.train.batch_views,))
            batch = [samples[i] for i in idx]
            m = trainer.train_step(batch)
            if (it + 1) % max(1, iters // 5) == 0:
                print(f"  it {it+1}/{iters}  loss={m['loss']:.5f}  "
                      f"photo={m['photo']:.5f}  rel_l2={m['rel_l2']:.5f}")
        trainer.save(tag=stage)

    # ---- Phase 4: evaluation ----------------------------------------- #
    print(f"\n[{category}] evaluating...")
    evaluator = Evaluator(cfg, device=device)
    eval_ds = DNeRFDataset(cfg, split="test", categories=[category])
    eval_samples = [eval_ds[i] for i in range(
        min(len(eval_ds), args.max_eval))]
    if eval_samples:
        metrics = evaluator.evaluate(trainer.model, eval_samples)
        print(f"[{category}] PSNR={metrics.psnr_mean:.3f}  "
              f"SSIM={metrics.ssim_mean:.4f}  "
              f"LPIPS={metrics.lpips_mean:.4f}  "
              f"(n={metrics.n_samples})")
    else:
        metrics = None
        print(f"[{category}] no test samples; skipping eval")

    stability = evaluator.rollout_stability(
        trainer.model, n_steps=args.rollout_steps)
    final_drift = stability.pos_drift[-1] if stability.pos_drift else 0.0
    print(f"[{category}] rollout drift @ {args.rollout_steps} steps: "
          f"{final_drift:.5f}")

    # Save results.
    results = {
        "category": category,
        "static_gaussians": cloud.n,
        "eval": {"psnr": metrics.psnr_mean if metrics else None,
                  "ssim": metrics.ssim_mean if metrics else None,
                  "lpips": metrics.lpips_mean if metrics else None}
            if metrics else None,
        "rollout_final_drift": final_drift,
        "rollout_steps": stability.steps,
        "rollout_pos_drift": stability.pos_drift,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--category", default=None,
                   help="single D-NeRF category (default: all)")
    p.add_argument("--all", action="store_true", help="process all categories")
    p.add_argument("--out", default="./runs", help="output directory")
    p.add_argument("--iterations", type=int, default=1000,
                   help="training iterations per stage")
    p.add_argument("--static-iters", type=int, default=500,
                   help="static 3DGS init iterations")
    p.add_argument("--num-gaussians", type=int, default=None,
                   help="static 3DGS cloud size (default: cfg). The pure-torch "
                        "rasterizer is O(N*H*W) in autograd memory; on Apple "
                        "Silicon keep this <= 5000 to avoid swapping.")
    p.add_argument("--render-wh", type=int, nargs=2, default=None,
                   metavar=("W", "H"),
                   help="render resolution for differentiable supervision "
                        "(default: cfg.data.render_wh). Lower this to bound "
                        "memory on CPU/MPS.")
    p.add_argument("--time-window", type=int, default=None,
                   help="autoregressive rollout length per training step "
                        "(default: cfg). Each step renders T frames, so memory "
                        "scales linearly with this.")
    p.add_argument("--max-samples", type=int, default=200,
                   help="max training frames to load")
    p.add_argument("--max-eval", type=int, default=50,
                   help="max test frames to evaluate")
    p.add_argument("--rollout-steps", type=int, default=100,
                   help="autoregressive rollout length for stability test")
    p.add_argument("--stage", default=None,
                   choices=["dense", "farfield", "joint"],
                   help="train a single stage only")
    p.add_argument("--skip-static", action="store_true",
                   help="reuse existing static cloud if present")
    p.add_argument("--quick", action="store_true",
                   help="fast smoke run (few iters, few samples)")
    args = p.parse_args()

    if args.quick:
        args.iterations = 20
        args.static_iters = 20
        args.max_samples = 16
        args.max_eval = 8
        args.rollout_steps = 10

    cfg = Config()
    # Apply CLI memory overrides (the pure-torch rasterizer is O(N*H*W) in
    # autograd memory; defaults target a production-scale cloud that will swap
    # a laptop to death — see rules.md Rule 6).
    if args.num_gaussians is not None:
        cfg.static_gs.num_gaussians = args.num_gaussians
    if args.render_wh is not None:
        cfg.data.render_wh = tuple(args.render_wh)
    if args.time_window is not None:
        cfg.train.time_window = args.time_window
    if args.category:
        cfg.data.categories = (args.category,)
        cats = [args.category]
    elif args.all:
        cats = list(cfg.data.categories)
    else:
        cats = list(cfg.data.categories)

    all_results = {}
    for cat in cats:
        try:
            all_results[cat] = run_category(cfg, cat, args)
        except Exception as e:
            print(f"[{cat}] FAILED: {e}")
            import traceback; traceback.print_exc()
            all_results[cat] = {"error": str(e)}

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for cat, r in all_results.items():
        if "error" in r:
            print(f"  {cat:16s} ERROR: {r['error']}")
        else:
            ev = r.get("eval") or {}
            print(f"  {cat:16s} PSNR={ev.get('psnr', 0):.2f}  "
                  f"drift={r['rollout_final_drift']:.4f}")
    return all_results


if __name__ == "__main__":
    main()
