"""Loss functions.

Implements the supervision from workflow.md Phase 3 Step 8 (L1 + SSIM) plus
the autoregressive error mitigation from rules.md Rule 4 (relative L2 with
Gaussian noise injection)."""

from .rendering import L1SSIMLoss
from .pde import RelativeL2Loss, NoiseInjector

__all__ = ["L1SSIMLoss", "RelativeL2Loss", "NoiseInjector"]
