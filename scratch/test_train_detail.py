"""Detailed loss monitor to isolate NaN component."""
import torch
from mcrah.config import Config
from mcrah.training import load_cloud
from mcrah.training.trainer import MCRAHTrainer
from mcrah.data import SceneSample

def main():
    cfg = Config.for_category("trex")
    cfg.train.device = "cpu"
    cfg.hypergraph.adaptive = True
    
    cloud = load_cloud("runs/trex/static_cloud.pt", device="cpu")
    trainer = MCRAHTrainer(cfg, cloud, out_dir="scratch/test_train_detail")
    
    H, W = 400, 400
    samples = []
    for i, t_val in enumerate([0.0, 0.05, 0.10, 0.15, 0.20]):
        img = torch.rand(3, H, W)
        c2w = torch.eye(4)
        K = torch.tensor([[500.0, 0, 200.0], [0, 500.0, 200.0], [0, 0, 1.0]])
        samples.append(SceneSample("trex", t_val, i, img, c2w, K))
        
    trainer.set_stage("dense", total_steps=200)
    for it in range(1, 201):
        m = trainer.train_step(samples)
        nan_keys = [k for k, v in m.items() if isinstance(v, float) and torch.isnan(torch.tensor(v))]
        if nan_keys or it % 20 == 0:
            print(f"Iter {it:3d}: " + " ".join([f"{k}={v:.5f}" if not torch.isnan(torch.tensor(v)) else f"{k}=NaN" for k, v in m.items()]))
        if nan_keys:
            print(f"!!! NaN detected in components: {nan_keys}")
            break

if __name__ == "__main__":
    main()
