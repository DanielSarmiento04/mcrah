"""Reproduce training loop NaN issue."""
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
    trainer = MCRAHTrainer(cfg, cloud, out_dir="scratch/test_train")
    
    N = cloud.n
    H, W = 400, 400
    
    # Create dummy samples
    samples = []
    for i, t_val in enumerate([0.0, 0.05, 0.10, 0.15, 0.20]):
        img = torch.rand(3, H, W)
        c2w = torch.eye(4)
        K = torch.tensor([[500.0, 0, 200.0], [0, 500.0, 200.0], [0, 0, 1.0]])
        samples.append(SceneSample("trex", t_val, i, img, c2w, K))
        
    print("Starting train steps...")
    trainer.set_stage("dense", total_steps=100)
    for it in range(1, 101):
        m = trainer.train_step(samples)
        if it % 10 == 0 or torch.isnan(torch.tensor(m["loss"])):
            print(f"it {it}: loss={m['loss']:.5f} photo={m['photo']:.5f} rig={m['rigidity']:.5f} topo={m['topology']:.5f} pde={m['pde']:.5f} rel_l2={m['rel_l2']:.5f} prior_logit={trainer.model.mcrah.gate.prior_logit.item():.4f}")
        if torch.isnan(torch.tensor(m["loss"])):
            print("NaN detected!")
            break

if __name__ == "__main__":
    main()
