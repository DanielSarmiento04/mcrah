"""Test rollout stability for 100 steps to reproduce NaN drift."""
import torch
from mcrah.config import Config
from mcrah.training import load_cloud, Evaluator
from mcrah.training.trainer import MCRAHTrainer

def test():
    cfg = Config.for_category("trex")
    cfg.train.device = "cpu"
    cloud = load_cloud("runs/trex/static_cloud.pt", device="cpu")
    trainer = MCRAHTrainer(cfg, cloud, out_dir="scratch/test_stab")
    model = trainer.model
    model.eval()
    
    evaluator = Evaluator(cfg, device="cpu")
    print("Running rollout stability for 100 steps...")
    stab = evaluator.rollout_stability(model, n_steps=100)
    print("Steps completed:", len(stab.steps))
    for i in [0, 10, 20, 50, 80, 99]:
        if i < len(stab.pos_drift):
            print(f"Step {i}: pos_drift={stab.pos_drift[i]}, rot_drift={stab.rot_drift[i]}")

if __name__ == "__main__":
    test()
