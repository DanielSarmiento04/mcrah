"""Locate the first non-finite tensor in the MCRAH training forward pass."""
import sys
import torch
from mcrah.config import Config
from mcrah.training import load_cloud
from mcrah.training.trainer import MCRAHTrainer


def stats(name, t):
    if t is None:
        return
    t = t.detach()
    fin = torch.isfinite(t)
    tf = t[fin].float()
    print(f"  {name:28s} dtype={str(t.dtype):14s} nonfinite={int((~fin).sum()):7d}"
          f"  absmax={float(tf.abs().max()) if tf.numel() else float('nan'):.4g}")


def main(amp_dtype=None):
    cfg = Config.for_category("trex") if hasattr(Config, "for_category") else Config()
    cloud = load_cloud("runs/trex/static_cloud.pt", device="cpu")
    print("== static cloud ==")
    for k in ("means", "scales", "rotations", "opacities", "sh"):
        stats(k, getattr(cloud, k))

    # Force CPU so we can run locally.
    cfg.train.device = "cpu"
    trainer = MCRAHTrainer(cfg, cloud, out_dir="scratch/diag_run")
    model = trainer.model
    print("device:", trainer.device, "adaptive:", cfg.hypergraph.adaptive,
          "E:", model.hypergraph.n_edges)

    # Hooks on every submodule output.
    first_bad = []

    def hook(mod, inp, out, name=None):
        outs = out if isinstance(out, (tuple, list)) else (out,)
        for o in outs:
            if torch.is_tensor(o) and not torch.isfinite(o).all() and not first_bad:
                first_bad.append(name)
                print(f"  !! first non-finite output at module: {name} ({type(mod).__name__})")
                for i in (inp if isinstance(inp, tuple) else (inp,)):
                    if torch.is_tensor(i):
                        stats("   input", i)
                stats("   output", o)

    for n, m in model.named_modules():
        m.register_forward_hook(lambda mod, i, o, n=n: hook(mod, i, o, n))

    times = [torch.tensor(t) for t in (0.0, 0.05, 0.1, 0.15)]
    ctx = (torch.autocast("cpu", dtype=amp_dtype) if amp_dtype
           else torch.autocast("cpu", enabled=False))
    model.train()
    with ctx:
        steps = model.rollout(times, noise_injector=trainer.noise)
        print("== rollout ==")
        for i, s in enumerate(steps):
            stats(f"step{i}.delta_pos", s.delta_pos)
            stats(f"step{i}.delta_rot", s.delta_rot)
            stats(f"step{i}.membership", s.membership)
            stats(f"step{i}.cloud.means", s.cloud.means)
            if s.membership is not None:
                stats(f"step{i}.M.colsum", s.membership.float().sum(0))


if __name__ == "__main__":
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(
        sys.argv[1] if len(sys.argv) > 1 else "", None)
    print("AMP dtype:", dt)
    main(dt)
