"""Test safe axis_angle_to_quaternion implementation."""
import torch

def safe_axis_angle_to_quaternion(rotvec: torch.Tensor) -> torch.Tensor:
    """Convert (N,3) axis-angle to (N,4) quaternion (w,x,y,z), normalized.
    Numerically safe under FP16/AMP and float32 for arbitrarily small angles.
    """
    orig_dtype = rotvec.dtype
    r = rotvec.float()  # compute in float32 for AMP stability
    angle_sq = (r ** 2).sum(dim=-1, keepdim=True)
    
    # Safe branchless calculation using torch.where on smooth scale factor
    # sin(x/2)/x ~ 0.5 - x^2/48
    small_mask = angle_sq < 1e-8
    
    # Large angle calculation
    angle = torch.sqrt(angle_sq.clamp_min(1e-8))
    scale_large = torch.sin(0.5 * angle) / angle
    cos_large = torch.cos(0.5 * angle)
    
    # Small angle calculation (Taylor series)
    scale_small = 0.5 - angle_sq / 48.0
    cos_small = 1.0 - angle_sq / 8.0
    
    scale = torch.where(small_mask, scale_small, scale_large)
    cos = torch.where(small_mask, cos_small, cos_large)
    
    q = torch.cat([cos, r * scale], dim=-1)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return q.to(orig_dtype)

def test_safe_grad():
    print("Testing 0.0 input in FP16:")
    r1 = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float16, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.float16):
        q1 = safe_axis_angle_to_quaternion(r1)
        l1 = q1.sum()
    l1.backward()
    print("r1.grad:", r1.grad, "has_nan:", torch.isnan(r1.grad).any().item())

    print("\nTesting 1e-8 input in FP16:")
    r2 = torch.tensor([[1e-8, 1e-8, 1e-8]], dtype=torch.float16, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.float16):
        q2 = safe_axis_angle_to_quaternion(r2)
        l2 = q2.sum()
    l2.backward()
    print("r2.grad:", r2.grad, "has_nan:", torch.isnan(r2.grad).any().item())

if __name__ == "__main__":
    test_safe_grad()
