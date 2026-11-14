"""Differentiable 1 - SSIM structural loss computed in forced float32."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.registry import LOSS_REGISTRY


def _build_gaussian_window(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2.0
    gauss_1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    kernel_2d = torch.outer(gauss_1d, gauss_1d)
    return kernel_2d.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1).contiguous()


@LOSS_REGISTRY.register("ssim")
class SSIMLoss(nn.Module):
    """Differentiable `1.0 - SSIM(pred, target)` using an 11x11 Gaussian window."""

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
        channels: int = 3,
    ) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.sigma = float(sigma)
        self.data_range = float(data_range)
        self.channels = int(channels)
        self.register_buffer(
            "window",
            _build_gaussian_window(self.window_size, self.sigma, self.channels),
            persistent=False,
        )

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type=pred.device.type, enabled=False):
            p = pred.float()
            t = target.float()
            ch = p.shape[1]
            if ch == self.channels:
                win = self.window.to(device=p.device, dtype=torch.float32)
            else:
                win = _build_gaussian_window(self.window_size, self.sigma, ch).to(
                    device=p.device, dtype=torch.float32
                )
            pad = self.window_size // 2

            mu1 = F.conv2d(p, win, padding=pad, groups=ch)
            mu2 = F.conv2d(t, win, padding=pad, groups=ch)
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2

            sigma1_sq = (F.conv2d(p * p, win, padding=pad, groups=ch) - mu1_sq).clamp_min(0.0)
            sigma2_sq = (F.conv2d(t * t, win, padding=pad, groups=ch) - mu2_sq).clamp_min(0.0)
            sigma12 = F.conv2d(p * t, win, padding=pad, groups=ch) - mu1_mu2

            c1 = (0.01 * self.data_range) ** 2
            c2 = (0.03 * self.data_range) ** 2

            numerator = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
            denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
            ssim_map = numerator / denominator.clamp_min(1.0e-12)
            return 1.0 - ssim_map.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.compute(pred, target)
