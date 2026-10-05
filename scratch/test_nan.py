import torch
import torch.nn as nn
from mcrah.config import Config
from mcrah.gs.gaussian import GaussianCloud
from mcrah.models import MCRAH
from mcrah.training.trainer import MCRAHTrainer
from mcrah.data import SceneSample

def test():
    cfg = Config.for_category("trex")
    N = 19486
    E = 139
    cloud = GaussianCloud(
        means=torch.randn(N, 3),
        scales=torch.full((N, 3), -5.0),
        rotations=torch.randn(N, 4),
        opacities=torch.full((N, 1), 0.0),
        sh=torch.randn(N, 3),
    )
    trainer = MCRAHTrainer(cfg, cloud)
    print("Trainer built successfully.")
    
    # Check MCRAHGate
    gate = trainer.model.mcrah.gate
    print("prior_logit:", gate.prior_logit)
    print("prior_logit grad_fn/requires_grad:", gate.prior_logit.requires_grad)

    # Let's inspect _compute_loss
    T = 4
    H, W = 400, 400
    dev = trainer.device
    pred = torch.rand(T, 3, H, W, device=dev)
    target = torch.rand(T, 3, H, W, device=dev)
    deltas = [torch.randn(N, 3, device=dev) * 0.01 for _ in range(T)]
    
    # Steps
    times = [torch.tensor(i * 0.1, device=dev) for i in range(T)]
    steps = trainer.model.rollout(times)
    
    loss, metrics = trainer._compute_loss(pred, target, deltas, trainer.model.cloud, steps)
    print("Metrics:", metrics)
    print("Loss:", loss)
    
    # Try backward
    loss.backward()
    print("prior_logit grad:", gate.prior_logit.grad)

if __name__ == "__main__":
    test()
