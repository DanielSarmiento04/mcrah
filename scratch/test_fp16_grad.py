"""Test backward pass of axis_angle_to_quaternion with FP16 small values."""
import torch
from mcrah.models.simgnn import axis_angle_to_quaternion

def test_grad_fp16():
    rotvec = torch.tensor([[1e-8, 1e-8, 1e-8]], dtype=torch.float16, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.float16):
        q = axis_angle_to_quaternion(rotvec)
        loss = q.sum()
    loss.backward()
    print("rotvec.grad with 1e-8 input in FP16:")
    print(rotvec.grad)
    print("Has NaN:", torch.isnan(rotvec.grad).any().item())

if __name__ == "__main__":
    test_grad_fp16()
