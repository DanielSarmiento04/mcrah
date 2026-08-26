"""Training and evaluation infrastructure.

  * :class:`StaticGSInit` - Phase 1 Step 3: static 3DGS initialization at t=0.
  * :class:`MCRAHTrainer` - Phase 3 Steps 8-9: decoupled two-stage training.
  * :class:`Evaluator`     - Phase 4 Step 10: NVS metrics (PSNR/SSIM/LPIPS) +
    autoregressive rollout stability testing.
"""

from .static_init import StaticGSInit, StaticInitResult, save_cloud, load_cloud
from .trainer import MCRAHTrainer, TrainState
from .evaluator import Evaluator, EvalMetrics, RolloutStability, AblationResult, build_ablation_configs

__all__ = [
    "StaticGSInit", "StaticInitResult", "save_cloud", "load_cloud",
    "MCRAHTrainer", "TrainState",
    "Evaluator", "EvalMetrics", "RolloutStability", "AblationResult",
    "build_ablation_configs",
]
