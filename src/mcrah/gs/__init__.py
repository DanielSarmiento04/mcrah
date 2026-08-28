"""3D Gaussian Splatting utilities and differentiable rendering bridge.

This module provides the Gaussian primitive bookkeeping and a differentiable
rasterizer abstraction. The real rasterization backend (e.g. diff-gaussian-
rasterization or a custom Metal kernel) is plugged in via
:func:`set_rasterizer`; a pure-torch fallback (depth-sorted alpha compositing)
is provided so the full pipeline runs on Apple Silicon / MPS without the
CUDA-only reference implementation.
"""

from .gaussian import GaussianCloud, transform_gaussians, apply_offsets
from .rasterizer import (
    render, set_rasterizer, get_rasterizer,
    Rasterizer, PyTorchRasterizer,
    CUDAGaussianRasterizer,
    _cuda_rasterizer_available,
)

__all__ = [
    "GaussianCloud",
    "transform_gaussians",
    "apply_offsets",
    "render",
    "set_rasterizer",
    "get_rasterizer",
    "Rasterizer",
    "PyTorchRasterizer",
    "CUDAGaussianRasterizer",
]
