"""Test matmul overflow with N=20000 in FP16."""
import torch
import torch.nn.functional as F
from mcrah.models.mcrah import AdaptiveHypergraph

def test_fp16_overflow():
    N = 20000
    E = 100
    D = 64
    
    assignment_0 = torch.randint(0, E, (N,))
    hg = AdaptiveHypergraph(assignment_0, E, adaptive=True)
    
    # Soft membership M (N, E)
    M = torch.rand(N, E)
    M = F.softmax(M, dim=-1)
    hg.set_membership(M)
    
    x = torch.randn(N, D) * 2.0  # typical feature norm in deep networks
    
    # FP32 propagation
    out_fp32 = hg.propagate(x)
    print("FP32 max:", out_fp32.abs().max().item(), "has_nan:", torch.isnan(out_fp32).any().item(), "has_inf:", torch.isinf(out_fp32).any().item())
    
    # FP16 propagation
    with torch.autocast("cpu", dtype=torch.float16):
        out_fp16 = hg.propagate(x)
        print("FP16 max:", out_fp16.abs().max().item(), "has_nan:", torch.isnan(out_fp16).any().item(), "has_inf:", torch.isinf(out_fp16).any().item())

if __name__ == "__main__":
    test_fp16_overflow()
