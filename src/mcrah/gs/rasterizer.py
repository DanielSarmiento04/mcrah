"""Differentiable Gaussian-splatting rasterization bridge.

The reference 3DGS rasterizer (``diff-gaussian-rasterization``) is CUDA-only and
therefore cannot run on the Apple M3 Pro / MPS prototyping target (rules.md
Rule 6). This module provides:

  * :class:`Rasterizer`       - backend-agnostic abstract interface.
  * :class:`PyTorchRasterizer` - a pure-torch EWA-splatting fallback that does
    depth-sorted front-to-back alpha compositing. It is fully differentiable
    and runs anywhere PyTorch does (CPU/MPS/CUDA), so the full MCRAH pipeline
    can be debugged end-to-end before the CUDA backend is plugged in.
  * :class:`CUDAGaussianRasterizer` - a thin adapter over the official
    ``diff_gaussian_rasterization`` package, lazily imported so the dependency
    is optional.

The active backend is selected with :func:`set_rasterizer` and invoked through
the module-level :func:`render` entry point, keeping call sites backend-agnostic.

Rendering math (EWA splatting, Zwicker et al. 2001 / Kerbl et al. 2023):
  1. Activate the cloud (exp scales, sigmoid opacity, normalize quaternion).
  2. World -> camera via ``W = R_c2w^T`` (rigid inverse), ``t = -W @ p``.
  3. Build 3D covariance ``Sigma = R_g diag(s^2) R_g^T`` in world space.
  4. Transform to camera space ``Sigma_cam = W Sigma W^T``.
  5. Perspective Jacobian ``J`` maps camera-space -> screen pixel offsets.
  6. 2D covariance ``Sigma_2d = J Sigma_cam J^T``; conic = its inverse.
  7. Splat each Gaussian onto its 3-sigma bounding box, composite front-to-back:
        C += a * T * color ;  T *= (1 - a) ;  a = opacity * exp(-0.5 d^T conic d)
     stopping when transmittance ``T`` drops below a threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from .gaussian import GaussianCloud, rotation_to_matrix


# --------------------------------------------------------------------------- #
# Output type
# --------------------------------------------------------------------------- #
@dataclass
class RenderOutput:
    """Result of rasterizing one view of a Gaussian cloud."""
    image: torch.Tensor   # (3, H, W) float32 in [0,1]
    alpha: torch.Tensor   # (1, H, W) float32 accumulated opacity
    depth: torch.Tensor   # (1, H, W) float32, z-buffer of front surface
    n_visible: int        # number of Gaussians that contributed >=1 pixel


# --------------------------------------------------------------------------- #
# Backend abstraction
# --------------------------------------------------------------------------- #
class Rasterizer:
    """Abstract rasterizer backend."""

    def render(
        self,
        cloud: GaussianCloud,
        c2w: torch.Tensor,
        intrinsics: torch.Tensor,
        width: int,
        height: int,
        bg_color: Optional[torch.Tensor] = None,
    ) -> RenderOutput:  # pragma: no cover - interface
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Pure-torch EWA splatting fallback (MPS-compatible)
# --------------------------------------------------------------------------- #
class PyTorchRasterizer(Rasterizer):
    """Depth-sorted front-to-back alpha-compositing EWA splatter.

    This is intentionally a *reference* implementation: each Gaussian is splatted
    onto its own bounding box and composited in depth order. The compositing is
    sequential over Gaussians (front-to-back is inherently a scan), so it is
    correct but slower than a tiled CUDA kernel. It is the right choice for
    prototyping on Apple Silicon where no CUDA backend exists.
    """

    def __init__(
        self,
        max_patch: int = 64,      # cap on the half-extent of a splat (px)
        min_patch: int = 1,
        eps2d: float = 0.3,       # anti-singularity added to the 2D covariance
        early_out: float = 1e-3,  # stop compositing a pixel once T < early_out
    ):
        self.max_patch = max_patch
        self.min_patch = min_patch
        self.eps2d = eps2d
        self.early_out = early_out

    def _project(self, g: GaussianCloud, c2w: torch.Tensor, K: torch.Tensor,
                 width: int, height: int):
        """Project Gaussians to screen space; return splat params + valid mask.

        Differentiable: the EWA projection math (covariance, conic, power) all
        carry gradients so that the rendered image supervises the predicted
        Gaussian offsets (means, rotations). Only the *ordering* (argsort) and
        integer bounding boxes are detached, which is correct — depth order has
        no meaningful gradient. Wrap the caller in ``torch.no_grad()`` for
        inference to recover the fast non-tracking path.
        """
        device = g.means.device
        N = g.n
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # World -> camera (c2w is rigid OpenCV camera-to-world).
        W = c2w[:3, :3].transpose(0, 1)            # (3,3) rotation w2c
        t = -W @ c2w[:3, 3]                        # (3,) translation w2c
        mu_cam = g.means @ W.t() + t               # (N,3)

        z = mu_cam[:, 2]
        # Keep only Gaussians in front of the camera and within a sane depth.
        front = z > 1e-4
        zc = z.clamp_min(1e-4)

        u = fx * mu_cam[:, 0] / zc + cx            # (N,)
        v = fy * mu_cam[:, 1] / zc + cy            # (N,)

        # 3D covariance in world space.
        Rg = rotation_to_matrix(g.rotations)       # (N,3,3)
        s = g.scales.clamp_min(1e-6)               # (N,3) activated (already exp)
        S = torch.diag_embed(s)                    # (N,3,3)
        Sigma = Rg @ S @ S.transpose(1, 2) @ Rg.transpose(1, 2)  # (N,3,3)

        # Camera-space covariance: Sigma_cam = W Sigma W^T.
        Wn = W.unsqueeze(0)                        # (1,3,3)
        Sigma_cam = Wn @ Sigma @ Wn.transpose(1, 2)  # (N,3,3)

        # Perspective Jacobian at each mean. Build out-of-place so the graph
        # from g.means -> mu_cam -> J -> Sigma2d -> power -> alpha stays intact.
        x, y = mu_cam[:, 0], mu_cam[:, 1]
        z2 = zc * zc
        zeros = torch.zeros_like(zc)
        # J: (N, 2, 3). Row 0 = d(u)/d(cam), Row 1 = d(v)/d(cam).
        j_row0 = torch.stack([fx / zc, zeros, -fx * x / z2], dim=-1)  # (N,3)
        j_row1 = torch.stack([zeros, fy / zc, -fy * y / z2], dim=-1)  # (N,3)
        J = torch.stack([j_row0, j_row1], dim=1)  # (N,2,3)

        Sigma2d = J @ Sigma_cam @ J.transpose(1, 2)  # (N,2,2)
        # Regularize for invertibility / numerical stability.
        eye2 = torch.eye(2, device=device, dtype=g.means.dtype) * self.eps2d
        Sigma2d = Sigma2d + eye2

        # Conic (inverse covariance) for the power evaluation.
        a = Sigma2d[:, 0, 0]
        b = Sigma2d[:, 0, 1]
        c = Sigma2d[:, 1, 1]
        det = (a * c - b * b).clamp_min(1e-12)
        inv_det = 1.0 / det
        conic = torch.stack([
            c * inv_det, -b * inv_det, a * inv_det
        ], dim=-1)  # (N,3): [A, B, C] of inverse [[A,B],[B,C]]

        # Bounding-box half-extent from the 3-sigma iso-contour.
        # Eigenvalues of Sigma2d: lam = 0.5*(tr +/- sqrt((a-c)^2 + 4b^2)).
        tr = a + c
        disc = ((a - c) ** 2 + 4 * b * b).clamp_min(0.0).sqrt()
        lam_max = 0.5 * (tr + disc)            # larger variance
        radius = 3.0 * lam_max.clamp_min(1e-8).sqrt()   # 3-sigma
        radius = radius.clamp(self.min_patch, self.max_patch)

        # Cull anything clearly off-screen or behind the camera.
        r_int = radius.ceil().long()
        on_screen = front & (u + r_int >= 0) & (u - r_int < width) & \
            (v + r_int >= 0) & (v - r_int < height)

        return dict(
            u=u, v=v, z=z, color=g.sh, opacity=g.opacities[:, 0],
            conic=conic, radius=radius, r_int=r_int, valid=on_screen,
        )

    def render(
        self,
        cloud: GaussianCloud,
        c2w: torch.Tensor,
        intrinsics: torch.Tensor,
        width: int,
        height: int,
        bg_color: Optional[torch.Tensor] = None,
    ) -> RenderOutput:
        device = cloud.means.device
        dtype = cloud.means.dtype

        # Activate (exp scale, sigmoid opacity, normalize rotation).
        g = cloud.activated()

        p = self._project(g, c2w.to(device=device, dtype=dtype),
                          intrinsics.to(device=device, dtype=dtype), width, height)
        valid = p["valid"]
        if valid.sum() == 0:
            img = torch.zeros(3, height, width, device=device, dtype=dtype)
            if bg_color is not None:
                img = img + bg_color.to(device=device, dtype=dtype).view(3, 1, 1)
            return RenderOutput(image=img, alpha=torch.zeros(1, height, width,
                                device=device, dtype=dtype),
                                depth=torch.zeros(1, height, width,
                                device=device, dtype=dtype), n_visible=0)

        # Sort front-to-back (nearest depth first) and keep only valid gaussians.
        idx_all = torch.where(valid)[0]
        order = idx_all[torch.argsort(p["z"][idx_all])]

        u = p["u"][order]
        vv = p["v"][order]
        z = p["z"][order]
        color = p["color"][order]          # (M,3)
        opacity = p["opacity"][order]      # (M,)
        conic = p["conic"][order]          # (M,3)
        r_int = p["r_int"][order]          # (M,)

        # Background.
        if bg_color is None:
            bg = torch.zeros(3, device=device, dtype=dtype)
        else:
            bg = bg_color.to(device=device, dtype=dtype).view(3)

        # Accumulators. Out-of-place updates so autograd can flow from the
        # rendered image back through alpha, power, and conic to the Gaussian
        # means/rotations (essential for differentiable supervision).
        #
        # rgb_acc starts at ZERO (not bg). The background is added at the end
        # weighted by the remaining transmittance T_final:
        #   C = Σ(aᵢ·Tᵢ·colorᵢ) + T_final · bg
        # Starting rgb_acc at bg would compute C = bg + Σ(aᵢ·Tᵢ·colorᵢ), which
        # overflows past 1.0 when bg is white (1.0) and gets clamped — zeroing
        # all gradients (the flat-loss bug).
        rgb_acc = torch.zeros(3, height, width, device=device, dtype=dtype)
        T = torch.ones(1, height, width, device=device, dtype=dtype)
        depth_acc = torch.zeros(1, height, width, device=device, dtype=dtype)
        n_visible = 0

        # Front-to-back compositing, one Gaussian at a time.
        for m in range(order.shape[0]):
            r = int(r_int[m].item())
            if r < 1:
                r = 1
            # Keep the splat center as a differentiable tensor so the rendered
            # image supervises Gaussian *positions* (means) directly. Only the
            # integer bounding box (floor/ceil) is detached -- it is inherently
            # piecewise-constant -- matching the differentiability contract in
            # this module's docstring. Detaching via .item() here would erase
            # the primary means->image gradient path.
            um = u[m]
            vm = vv[m]
            umf = float(um.detach().item())
            vmf = float(vm.detach().item())
            x0 = max(int(math.floor(umf - r)), 0)
            x1 = min(int(math.ceil(umf + r)) + 1, width)
            y0 = max(int(math.floor(vmf - r)), 0)
            y1 = min(int(math.ceil(vmf + r)) + 1, height)
            if x1 <= x0 or y1 <= y0:
                continue

            xs = torch.arange(x0, x1, device=device, dtype=dtype) - um
            ys = torch.arange(y0, y1, device=device, dtype=dtype) - vm
            # power = exp(-0.5 * (A*dx^2 + 2*B*dx*dy + C*dy^2))
            A, B, C = conic[m, 0], conic[m, 1], conic[m, 2]
            # (h, w)
            dyy = (ys * ys)[:, None] * C
            dxx = (xs * xs)[None, :] * A
            dxy = (ys[:, None]) * (xs[None, :]) * (2.0 * B)
            power = torch.exp(-0.5 * (dxx + dxy + dyy)).clamp_max(1.0)

            alpha = (opacity[m] * power).clamp(0.0, 0.999999)  # (h,w)
            patch_T = T[:, y0:y1, x0:x1]                 # (1,h,w)
            visible = patch_T[0] > self.early_out
            if not bool(visible.any()):
                # No pixel left to update in this patch; remaining gaussians are
                # farther, but they might still hit other pixels -> keep looping.
                continue

            w = alpha * patch_T[0]                      # (h,w) contribution weight
            col = color[m].view(3, 1, 1)                # (3,1,1)
            # Out-of-place accumulation (keeps the graph intact).
            rgb_patch = rgb_acc[:, y0:y1, x0:x1] + w[None] * col
            rgb_acc = rgb_acc.clone()
            rgb_acc[:, y0:y1, x0:x1] = rgb_patch
            new_T = patch_T * (1.0 - alpha[None])
            T = T.clone()
            T[:, y0:y1, x0:x1] = new_T
            # Record front-surface depth (first opaque hit). Depth is diagnostic;
            # detached so it never contributes gradients.
            untouched = (depth_acc[:, y0:y1, x0:x1] == 0) & (w[None] > 1e-3)
            depth_patch = torch.where(
                untouched, z[m] * torch.ones_like(depth_acc[:, y0:y1, x0:x1]),
                depth_acc[:, y0:y1, x0:x1],
            )
            depth_acc = depth_acc.clone()
            depth_acc[:, y0:y1, x0:x1] = depth_patch
            n_visible += 1

        alpha_acc = 1.0 - T  # accumulated opacity
        # Background compositing: C = Σ(aᵢ·Tᵢ·colorᵢ) + T_final · bg.
        # The remaining transmittance T determines how much background shows
        # through. Without this, pixels with no Gaussians stay black instead
        # of showing the background.
        rgb_acc = rgb_acc + T * bg.view(3, 1, 1)
        return RenderOutput(image=rgb_acc.clamp(0.0, 1.0), alpha=alpha_acc,
                            depth=depth_acc, n_visible=n_visible)


# --------------------------------------------------------------------------- #
# Optional CUDA backend (diff-gaussian-rasterization)
# --------------------------------------------------------------------------- #
def _disable_autocast_ctx():
    """Return a context manager that disables autocast across PyTorch versions."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast("cuda", enabled=False)
        except Exception:
            pass
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        return torch.cuda.amp.autocast(enabled=False)
    import contextlib
    return contextlib.nullcontext()


class CUDAGaussianRasterizer(Rasterizer):
    """Adapter around the official ``diff_gaussian_rasterization`` package.

    Lazily imports the dependency so it remains optional. Used on cloud GPU
    nodes (agent.md: high-end cloud deployments) for production-speed training.

    Maintains standard 3DGS OpenCV convention (x-right, y-down, z-forward) with
    the perspective projection matrix matching the CUDA kernel's expected +Z.
    """

    def __init__(self, convert_to_cv: bool = False):
        self.convert_to_cv = convert_to_cv

    @staticmethod
    def _get_projection_matrix(
        znear: float, zfar: float, fovx: float, fovy: float,
        device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        """Standard 3DGS perspective projection matrix (OpenGL NDC [-1,1])."""
        tanHalfFovX = math.tan(fovx / 2)
        tanHalfFovY = math.tan(fovy / 2)
        P = torch.zeros(4, 4, device=device, dtype=dtype)
        P[0, 0] = 1.0 / tanHalfFovX
        P[1, 1] = 1.0 / tanHalfFovY
        P[3, 2] = 1.0
        P[2, 2] = zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P

    def render(
        self,
        cloud: GaussianCloud,
        c2w: torch.Tensor,
        intrinsics: torch.Tensor,
        width: int,
        height: int,
        bg_color: Optional[torch.Tensor] = None,
    ) -> RenderOutput:  # pragma: no cover - requires CUDA + optional dep
        try:
            from diff_gaussian_rasterization import (  # type: ignore
                GaussianRasterizationSettings, GaussianRasterizer as _CUDARas,
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "CUDAGaussianRasterizer requires the 'diff_gaussian_rasterization' "
                "package on a CUDA device.  Install it with:\n"
                "  pip install git+https://github.com/graphdeco-inria/"
                "diff-gaussian-rasterization\n"
                "or fall back to the pure-torch rasterizer with "
                "set_rasterizer('torch')."
            ) from e

        with _disable_autocast_ctx():
            device = cloud.means.device
            dtype = torch.float32

            g = GaussianCloud(
                means=cloud.means.to(device=device, dtype=dtype),
                scales=cloud.scales.to(device=device, dtype=dtype),
                rotations=cloud.rotations.to(device=device, dtype=dtype),
                opacities=cloud.opacities.to(device=device, dtype=dtype),
                sh=cloud.sh.to(device=device, dtype=dtype),
            ).activated()

            # World-to-camera from c2w (OpenCV convention: x-right, y-down, z-fwd).
            c2w_f = c2w.to(device=device, dtype=dtype)
            intrinsics_f = intrinsics.to(device=device, dtype=dtype)
            w2c = torch.inverse(c2w_f)

            if self.convert_to_cv:
                cv2gl = torch.diag(torch.tensor(
                    [1.0, -1.0, -1.0, 1.0], device=device, dtype=dtype))
                w2c = cv2gl @ w2c

            # Camera position in world space (= c2w translation, NOT w2c[:3,3]).
            campos = c2w_f[:3, 3].contiguous()

            fx, fy = float(intrinsics_f[0, 0].item()), float(intrinsics_f[1, 1].item())
            fovx = 2.0 * math.atan(width / (2.0 * fx))
            fovy = 2.0 * math.atan(height / (2.0 * fy))

            bg = (bg_color.to(device=device, dtype=dtype) if bg_color is not None
                  else torch.zeros(3, device=device, dtype=dtype)
                  ).contiguous()

            znear, zfar = 0.01, 100.0
            P = self._get_projection_matrix(znear, zfar, fovx, fovy, device, dtype)

            # Transpose for glm column-major (official 3DGS convention).
            # In official 3DGS, full_proj_transform = world_view_transform @ projection_matrix
            # where both world_view_transform (w2c^T) and projection_matrix (P^T) are column-major transposed.
            viewmatrix = w2c.transpose(0, 1).contiguous()
            projmatrix = (viewmatrix.unsqueeze(0).bmm(P.transpose(0, 1).unsqueeze(0))).squeeze(0).contiguous()

            raster_settings = GaussianRasterizationSettings(
                image_height=height, image_width=width,
                tanfovx=math.tan(fovx / 2), tanfovy=math.tan(fovy / 2),
                bg=bg, scale_modifier=1.0,
                viewmatrix=viewmatrix, projmatrix=projmatrix,
                sh_degree=0, campos=campos,
                prefiltered=False, debug=False,
            )

            means3D = g.means.contiguous().float()
            means2D = torch.zeros_like(means3D, dtype=torch.float32)
            opacity = g.opacities.contiguous().float()
            colors = g.sh.contiguous().float()              # (N,3) DC RGB
            scales = g.scales.contiguous().float()          # (N, 3) activated
            rotations = g.rotations.contiguous().float()     # (N, 4) normalized

            raster = _CUDARas(raster_settings=raster_settings)
            img, radii = raster(
                means3D=means3D, means2D=means2D, shs=None,
                colors_precomp=colors, opacities=opacity,
                scales=scales, rotations=rotations,
            )
            return RenderOutput(
                image=img, alpha=torch.ones(1, height, width, device=device,
                                            dtype=dtype),
                depth=torch.zeros(1, height, width, device=device, dtype=dtype),
                n_visible=int((radii > 0).sum().item()),
            )


# --------------------------------------------------------------------------- #
# Registry / module-level entry point
# --------------------------------------------------------------------------- #
_RASTERIZER: Optional[Rasterizer] = None


def _cuda_rasterizer_available() -> bool:
    """Probe whether the diff_gaussian_rasterization package is importable
    AND a CUDA device is present.  The package is a CUDA C++ extension
    (not on PyPI) so it may be absent even on a CUDA machine."""
    if not torch.cuda.is_available():
        return False
    try:
        import diff_gaussian_rasterization  # noqa: F401
        return True
    except Exception:
        return False


def set_rasterizer(rasterizer: Rasterizer | str | None = "auto") -> Rasterizer:
    """Install a backend. ``"auto"`` -> CUDA if available else pure-torch,
    ``"torch"`` -> pure-torch fallback, ``"cuda"`` -> CUDA adapter, or pass a
    :class:`Rasterizer` instance directly.

    ``"auto"`` probes for the *actual import* of
    ``diff_gaussian_rasterization`` — not just ``torch.cuda.is_available()``
    — so the system gracefully falls back to the pure-torch rasterizer when
    the package is missing (common on Colab/free-tier GPU instances).
    """
    global _RASTERIZER
    if rasterizer == "auto" or rasterizer is None:
        if _cuda_rasterizer_available():
            _RASTERIZER = CUDAGaussianRasterizer()
        else:
            _RASTERIZER = PyTorchRasterizer()
    elif rasterizer == "torch":
        _RASTERIZER = PyTorchRasterizer()
    elif rasterizer == "cuda":
        _RASTERIZER = CUDAGaussianRasterizer()
    elif isinstance(rasterizer, Rasterizer):
        _RASTERIZER = rasterizer
    else:
        raise ValueError(f"unknown rasterizer: {rasterizer!r}")
    return _RASTERIZER


def get_rasterizer() -> Rasterizer:
    global _RASTERIZER
    if _RASTERIZER is None:
        _RASTERIZER = PyTorchRasterizer()
    return _RASTERIZER


def render(
    cloud: GaussianCloud,
    c2w: torch.Tensor,
    intrinsics: torch.Tensor,
    width: int,
    height: int,
    bg_color: Optional[torch.Tensor] = None,
) -> RenderOutput:
    """Render a Gaussian cloud from one camera using the active backend."""
    return get_rasterizer().render(cloud, c2w, intrinsics, width, height, bg_color)
