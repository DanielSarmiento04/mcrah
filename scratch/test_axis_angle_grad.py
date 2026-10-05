"""Test backward pass of axis_angle_to_quaternion for NaN gradients."""
import torch
from mcrah.models.simgnn import axis_angle_to_quaternion

def test_grad():
    # Zero rotvec (exact zero or very small)
    rotvec = torch.zeros(5, 3, requires_grad=True)
    q = axis_angle_to_quaternion(rotvec)
    loss = q.sum()
    loss.backward()
    print("rotvec.grad with zero input:")
    print(rotvec.grad)
    print("Has NaN:", torch.isnan(rotvec.grad).any().item())

if __name__ == "__main__":
    test_grad()
