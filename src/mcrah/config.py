"""Central configuration via dataclasses. Keeps every architectural knob
(rules.md) in one auditable place and mirrors the CVPR submission defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class DataConfig:
    data_root: Path = Path("data")
    processed_root: Path = Path("processed")
    categories: Tuple[str, ...] = (
        "bouncingballs",
        "hellwarrior",
        "hook",
        "jumpingjacks",
        "lego",
        "mutant",
        "standup",
        "trex",
    )
    # Image loading
    image_wh: Tuple[int, int] = (800, 800)  # D-NeRF native resolution (verified)
    channels: int = 3
    # Whether to read Rust-preprocessed .npy (False => read raw JSON+PNG)
    use_processed: bool = False
    # Augmentation
    white_background: bool = True  # D-NeRF renders on white bg
    num_workers: int = 4
    # Render resolution cap for differentiable supervision. The pure-torch
    # rasterizer clones the full image per Gaussian inside autograd, so rendering
    # at the native 800x800 with 50k Gaussians is O(N*H*W) memory and will swap
    # any laptop to death (rules.md Rule 6: Apple Silicon prototyping). This cap
    # down-samples the supervision resolution; targets are bilinearly resampled.
    render_wh: Tuple[int, int] = (200, 200)


@dataclass
class StaticGSConfig:
    """Phase 1 Step 3: static 3DGS initialization at t=0."""
    num_gaussians: int = 50_000
    sh_degree: int = 3
    density_init: float = 0.1
    iterations: int = 30_000
    lr_means: float = 1.6e-4
    lr_scales: float = 5e-3
    lr_opacity: float = 5e-2
    lr_sh: float = 2.5e-3


@dataclass
class HypergraphConfig:
    """Phase 2 Step 4/5: clustering + DHGC."""
    num_edges: int = 0  # 0 => auto sqrt(N)
    # Multi-scale dilation levels for far-field attention (rules.md Rule 2/3)
    dilation_levels: Tuple[int, ...] = (1, 2, 4)
    cosine_eps: float = 1e-4
    seed: int = 42
    # MCRAH: Motion-Coherent Rigidity-Adaptive Hypergraph (novel contribution).
    # When True, hyperedge membership evolves per autoregressive step via a
    # learned motion-coherence gate (see models/mcrah.py). When False, the
    # hypergraph is static (DVHGNN baseline / ablation arm).
    adaptive: bool = True
    mcrae_tau: float = 1.0           # membership softmax temperature


@dataclass
class MCRAHLossConfig:
    """Loss weights for the MCRAH novel-module regularizers."""
    rigidity_weight: float = 0.05    # quasi-rigid-body prior
    topology_smoothness_weight: float = 1e-3  # temporal membership stability


@dataclass
class ModelConfig:
    """MCRAH architecture."""
    feat_dim: int = 64
    hidden_dim: int = 256
    num_heads: int = 8
    num_simgnn_layers: int = 4
    num_dhgc_layers: int = 2
    # Fourier positional encoding frequencies (coordinates & time)
    num_freqs_pos: int = 6           # L_pos: 3D coordinates -> 3 + 6*6 = 39 dims
    num_freqs_time: int = 4          # L_time: time -> 1 + 2*4 = 9 dims
    # Gaussian attribute prediction heads & motion damping
    predict_offset: bool = True       # Delta position
    predict_rotation: bool = True     # Delta rotation (as quaternion delta)
    predict_scale: bool = False
    predict_sh: bool = False
    pos_scale: float = 0.20           # Max displacement scale in OffsetHeads (calibrated from 0.5)
    motion_damping: float = 0.95     # Residual motion damping factor
    # Noise injection (rules.md Rule 4 / agent.md directive 4)
    noise_std: float = 1e-2
    noise_warmup_steps: int = 2_000
    # PDE temporal smoothness weight (agent.md: Neural Physics Operators).
    pde_weight: float = 1e-2
    # Relative L2 regularization weight (rules.md Rule 4)
    rel_l2_weight: float = 1e-3
    # Activation
    dropout: float = 0.1


@dataclass
class TrainConfig:
    # Decoupled two-stage training (rules.md Rule 8 / workflow Phase 3 Step 9)
    stage: str = "dense"  # "dense" | "farfield" | "joint"
    batch_views: int = 4
    time_window: int = 8         # autoregressive rollout length per step
    rollout_gt_reset_prob: float = 0.1  # teacher forcing schedule
    iterations: int = 100_000
    lr: float = 1e-3
    lr_decay: float = 0.98
    # Stage-specific initial learning rates
    lr_dense: float = 1e-3
    lr_farfield: float = 5e-4
    lr_joint: float = 2e-4
    lr_min: float = 1e-6
    use_cosine_scheduler: bool = True
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    # Loss weights
    w_l1: float = 0.8
    w_ssim: float = 0.2
    w_rel_l2: float = 0.05
    w_lpips: float = 0.05  # perceptual loss weight (activated in joint stage)
    # Eval
    eval_every: int = 2_000
    save_every: int = 5_000
    # Device
    device: str = "auto"  # "auto" | "cuda" | "mps" | "cpu"
    # Hardware optimization
    use_amp: bool = True       # automatic mixed precision on CUDA (no-op else)
    # On CUDA with the official rasterizer, rendering at full 800x800 is fine;
    # on the pure-torch fallback, keep render_wh low to bound O(N*H*W) memory.
    cuda_render_wh: Tuple[int, int] = (800, 800)  # used only on CUDA backend


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    static_gs: StaticGSConfig = field(default_factory=StaticGSConfig)
    hypergraph: HypergraphConfig = field(default_factory=HypergraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    mcrah_loss: MCRAHLossConfig = field(default_factory=MCRAHLossConfig)

    @classmethod
    def for_category(cls, category: str) -> "Config":
        cfg = cls()
        cfg.data.categories = (category,)
        return cfg

    def device_str(self) -> str:
        if self.train.device != "auto":
            return self.train.device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def auto_render_wh(self) -> Tuple[int, int]:
        """Pick render resolution based on available hardware.

        On CUDA with the official diff-gaussian-rasterization, full 800x800
        is feasible.  On the pure-torch fallback (CPU/MPS or CUDA without
        the package), keep the capped resolution to bound O(N*H*W) memory.
        """
        dev = self.device_str()
        if dev == "cuda":
            try:
                import diff_gaussian_rasterization  # noqa: F401
                return self.train.cuda_render_wh
            except Exception:
                pass
        return self.data.render_wh
