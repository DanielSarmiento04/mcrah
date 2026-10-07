"""Pinpoint exact step and tensor that causes NaN in rollout_stability."""
import torch
from mcrah.config import Config
from mcrah.training import load_cloud
from mcrah.training.trainer import MCRAHTrainer

def main():
    cfg = Config.for_category("trex")
    cfg.train.device = "cpu"
    cloud = load_cloud("runs/trex/static_cloud.pt", device="cpu")
    trainer = MCRAHTrainer(cfg, cloud, out_dir="scratch/test_stab_pinpoint")
    model = trainer.model
    model.eval()

    n_steps = 100
    dt = 0.01
    dev = "cpu"

    base_means = model.cloud.means.detach().clone()
    base_rot = model.cloud.rotations.detach().clone()

    c = model.cloud
    for i in range(n_steps):
        t = torch.tensor(i * dt, device=dev)
        step = model.step(c, t)
        
        # Check step outputs
        if not torch.isfinite(step.delta_pos).all():
            print(f"Step {i}: delta_pos has non-finite values!")
            break
        if not torch.isfinite(step.delta_rot).all():
            print(f"Step {i}: delta_rot has non-finite values!")
            break
        if not torch.isfinite(step.cloud.means).all():
            print(f"Step {i}: cloud.means has non-finite values!")
            break
        if not torch.isfinite(step.cloud.rotations).all():
            print(f"Step {i}: cloud.rotations has non-finite values!")
            break
            
        dpos = (step.cloud.means - base_means).norm(dim=-1).mean().item()
        dot = (step.cloud.rotations * base_rot).sum(-1).abs().clamp_max(1.0)
        ang = 2.0 * torch.acos(dot.clamp(-1.0, 1.0)).mean().item()
        
        if torch.isnan(torch.tensor(dpos)):
            print(f"Step {i}: dpos is NaN! cloud.means max={step.cloud.means.abs().max()}")
            break
        if torch.isnan(torch.tensor(ang)):
            print(f"Step {i}: ang is NaN! dot min={dot.min()} max={dot.max()}")
            break
            
        c = step.cloud
        if i % 20 == 0 or i == n_steps - 1:
            print(f"Step {i:2d}: dpos={dpos:.6f}, ang={ang:.6f}, delta_pos_max={step.delta_pos.abs().max():.6f}")

if __name__ == "__main__":
    main()
