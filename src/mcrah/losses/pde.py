"""Autoregressive error-mitigation losses (rules.md Rule 4 / agent.md directive 4).

Combats long-term error drift via:
  * Relative L2 regularization on the predicted offsets, anchoring each
    predicted delta to the magnitude of the underlying state.
  * Gaussian noise injection during training to regularize the autoregressive
    rollout and improve robustness to accumulated state perturbations.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RelativeL2Loss(nn.Module):
    """Relative-L2 regularization on predicted offsets.

    Given a predicted offset ``d`` and the state it acts on ``x`` (e.g. a
    Gaussian mean or rotation), the loss is::

        ||d||^2 / (||x||^2 + eps)

    This keeps the regression from drifting in scale and is exactly the
    "relative L2 regularization" called out in rules.md Rule 4.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, delta: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # Match trailing dims: state may be (N,3), delta (N,3).
        if delta.shape != state.shape:
            state = state.expand_as(delta)
        num = (delta ** 2).sum(dim=-1, keepdim=True)
        den = (state ** 2).sum(dim=-1, keepdim=True) + self.eps
        return (num / den).mean()


class NoiseInjector(nn.Module):
    """Injects annealed Gaussian noise into the autoregressive state.

    During the dense-field propagation stage, the network predicts physical
    offsets autoregressively. To prevent the model from memorizing a single
    trajectory (and to combat error drift, agent.md directive 4), we add
    Gaussian noise to the *input* state at each rollout step, with a magnitude
    that decays over training:

        std(step) = noise_std * max(0, 1 - step/warmup)

    The noise is only active during training.
    """

    def __init__(self, std: float = 1e-2, warmup_steps: int = 2000):
        super().__init__()
        self.std = std
        self.warmup_steps = max(1, warmup_steps)
        self.register_buffer("global_step", torch.zeros(1, dtype=torch.long))

    @property
    def current_std(self) -> float:
        frac = float(self.global_step.item()) / self.warmup_steps
        frac = max(0.0, 1.0 - frac)
        return self.std * frac

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        std = self.current_std
        if std <= 0:
            return x
        return x + torch.randn_like(x) * std

    def step(self, n: int = 1) -> None:
        self.global_step += n


class PDERegularizer(nn.Module):
    """Lightweight PDE-style smoothness prior on the predicted deformation
    field (agent.md: Neural Physics Operators / PDE regularization).

    Penalizes the temporal second derivative of the predicted offsets across a
    rollout, approximating a damped-wave regularizer that discourages
    high-frequency temporal jitter (a key driver of autoregressive drift).
    """

    def __init__(self, lam: float = 1e-2):
        super().__init__()
        self.lam = lam

    def forward(self, offsets: torch.Tensor) -> torch.Tensor:
        # offsets: (T, N, D) autoregressive predictions over T timesteps.
        if offsets.shape[0] < 3:
            return offsets.new_zeros(())
        d2 = offsets[2:] - 2 * offsets[1:-1] + offsets[:-2]  # (T-2,N,D)
        return self.lam * (d2 ** 2).mean()
