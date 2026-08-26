"""Photometric rendering losses (L1 + D-SSIM)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(window_size: int, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def _ssim_map(
    x: torch.Tensor, y: torch.Tensor, window: torch.Tensor, c1: float, c2: float
) -> torch.Tensor:
    # x,y: (B,3,H,W)
    mu_x = F.conv2d(x, window, padding=window.shape[-1] // 2, groups=x.shape[1])
    mu_y = F.conv2d(y, window, padding=window.shape[-1] // 2, groups=y.shape[1])
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, window, padding=window.shape[-1] // 2, groups=x.shape[1]) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=window.shape[-1] // 2, groups=y.shape[1]) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=window.shape[-1] // 2, groups=x.shape[1]) - mu_xy
    ssim = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return ssim


class L1SSIMLoss(nn.Module):
    """Combined L1 + (1-SSIM) loss, the standard NVS photometric objective
    (workflow.md Phase 3 Step 8)."""

    def __init__(self, w_l1: float = 0.8, w_ssim: float = 0.2, window_size: int = 11):
        super().__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.window_size = window_size
        win1d = _gaussian_window(window_size)
        win2d = win1d[:, None] * win1d[None, :]
        self.register_buffer("_win", win2d[None, None])
        # broadcast the window across channels at forward time
        self.c1 = (0.01) ** 2
        self.c2 = (0.03) ** 2

    def _channel_window(self, c: int, device, dtype) -> torch.Tensor:
        win = self._win.to(device=device, dtype=dtype)
        return win.expand(c, 1, self.window_size, self.window_size).contiguous()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred,target: (B,3,H,W) in [0,1]
        l1 = F.l1_loss(pred, target)
        win = self._channel_window(pred.shape[1], pred.device, pred.dtype)
        ssim_map = _ssim_map(pred, target, win, self.c1, self.c2)
        ssim_loss = 1.0 - ssim_map.mean()
        return self.w_l1 * l1 + self.w_ssim * ssim_loss

    @torch.no_grad()
    def ssim_value(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        win = self._channel_window(pred.shape[1], pred.device, pred.dtype)
        return _ssim_map(pred, target, win, self.c1, self.c2).mean()


# Standard NVS metrics (Phase 4 Step 10).
@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


@torch.no_grad()
def ssim_metric(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    loss = L1SSIMLoss(window_size=window_size)
    return loss.ssim_value(pred, target)
