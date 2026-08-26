"""3D Gaussian primitive bookkeeping.

A Gaussian cloud stores the standard 3DGS attributes:
  * means       (N,3)
  * scales      (N,3)  (log-space inside the optimizer, activation outside)
  * rotations   (N,4)  quaternions (real-first)
  * opacities   (N,1)
  * sh coeffs   (N, K, 3)  spherical harmonics (K = (deg+1)^2); K=1 => DC color)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def rotation_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Quaternion (w,x,y,z) -> 3x3 rotation matrix. (N,4) -> (N,3,3)."""
    w, x, y, z = q.unbind(-1)
    n = torch.sqrt((q * q).sum(-1, keepdim=True).clamp_min(1e-12))
    w, x, y, z = w / n[..., 0], x / n[..., 0], y / n[..., 0], z / n[..., 0]
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        -1,
    ).reshape(*q.shape[:-1], 3, 3)


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of quaternions (w,x,y,z). (...,4) x (...,4) -> (...,4)."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        -1,
    )


def quaternion_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


@dataclass
class GaussianCloud:
    means: torch.Tensor       # (N,3)
    scales: torch.Tensor      # (N,3)  log-scale (pre-activation)
    rotations: torch.Tensor   # (N,4)  quaternion
    opacities: torch.Tensor    # (N,1)  logit (pre-activation)
    sh: torch.Tensor          # (N,3) DC term only (degree 0); extensible.

    @property
    def n(self) -> int:
        return self.means.shape[0]

    def activated(self) -> "GaussianCloud":
        return GaussianCloud(
            means=self.means,
            scales=torch.exp(self.scales).clamp_max(0.1),
            rotations=quaternion_normalize(self.rotations),
            opacities=torch.sigmoid(self.opacities),
            sh=self.sh,
        )

    def to(self, device) -> "GaussianCloud":
        return GaussianCloud(
            self.means.to(device), self.scales.to(device), self.rotations.to(device),
            self.opacities.to(device), self.sh.to(device),
        )

    def detach_clone(self) -> "GaussianCloud":
        return GaussianCloud(
            self.means.detach().clone(), self.scales.detach().clone(),
            self.rotations.detach().clone(), self.opacities.detach().clone(),
            self.sh.detach().clone(),
        )

    @classmethod
    def random(cls, n: int, device="cpu") -> "GaussianCloud":
        return cls(
            means=torch.randn(n, 3, device=device) * 0.5,
            scales=torch.full((n, 3), -2.0, device=device),
            rotations=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(n, 4).contiguous(),
            opacities=torch.zeros(n, 1, device=device),
            # Random DC color so the cloud is actually visible when rendered.
            # An all-black cloud composites to a constant-zero image against a
            # black background, which has a zero gradient w.r.t. every input.
            sh=torch.rand(n, 3, device=device),
        )


def transform_gaussians(cloud: GaussianCloud, c2w: torch.Tensor) -> GaussianCloud:
    """Apply a camera-to-world (or world-to-view) rigid transform to means/rot."""
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    new_means = cloud.means @ R.T + t
    Rmat = rotation_to_matrix(cloud.rotations)  # (N,3,3)
    # Rigid transform of the Gaussian orientation: R_new = R_world @ R_local.
    new_R = R.unsqueeze(0) @ Rmat
    # Convert back to quaternion via a simple branchless method.
    new_rot = _matrix_to_quaternion(new_R)
    return GaussianCloud(
        means=new_means, scales=cloud.scales, rotations=new_rot,
        opacities=cloud.opacities, sh=cloud.sh,
    )


def _matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """(N,3,3) rotation -> (N,4) quaternion (w,x,y,z). Robust Shepperd's method."""
    m = R
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    q = torch.zeros(*m.shape[:-2], 4, device=m.device, dtype=m.dtype)
    # branchless fallback: use trace + diagonals via where masks
    s = torch.sqrt((trace + 1.0).clamp_min(1e-12)) * 2
    q0 = torch.stack([
        0.25 * s,
        (m[..., 2, 1] - m[..., 1, 2]) / s,
        (m[..., 0, 2] - m[..., 2, 0]) / s,
        (m[..., 1, 0] - m[..., 0, 1]) / s,
    ], -1)
    q = torch.where(trace.unsqueeze(-1) > 0, q0, q)
    return quaternion_normalize(q)


def apply_offsets(
    cloud: GaussianCloud,
    delta_pos: torch.Tensor | None = None,
    delta_rot: torch.Tensor | None = None,
    delta_scale: torch.Tensor | None = None,
    delta_sh: torch.Tensor | None = None,
) -> GaussianCloud:
    """Apply predicted autoregressive offsets to a base Gaussian cloud
    (workflow.md Phase 3 Step 7). Rotations are composed as a delta quaternion."""
    means = cloud.means if delta_pos is None else cloud.means + delta_pos
    scales = cloud.scales if delta_scale is None else cloud.scales + delta_scale
    sh = cloud.sh if delta_sh is None else cloud.sh + delta_sh
    if delta_rot is not None:
        rot = quaternion_multiply(quaternion_normalize(cloud.rotations), delta_rot)
    else:
        rot = cloud.rotations
    return GaussianCloud(
        means=means, scales=scales, rotations=rot,
        opacities=cloud.opacities, sh=sh,
    )
