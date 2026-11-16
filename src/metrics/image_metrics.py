"""Pure PyTorch PSNR and SSIM evaluation metrics in float32."""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _ensure_4d_float32(tensor: torch.Tensor, data_range: float) -> torch.Tensor:
    """Convert `[C, H, W]` or `[B, C, H, W]` tensor to clamped float32 batch."""
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"Expected 3D or 4D tensor for metrics, got shape {tuple(tensor.shape)}")
    return tensor.detach().to(torch.float32).clamp(0.0, data_range)


@lru_cache(maxsize=16)
def _gaussian_window_cpu(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    """Create a normalized 2D Gaussian kernel shaped `(channels, 1, K, K)`."""
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2.0
    gauss_1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    kernel_2d = torch.outer(gauss_1d, gauss_1d)
    return kernel_2d.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1).contiguous()


@torch.no_grad()
def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    eps: float = 1.0e-12,
) -> float:
    """Compute mean batch PSNR (dB) on tensors clamped to `[0, data_range]`."""
    pred_f = _ensure_4d_float32(pred, data_range)
    target_f = _ensure_4d_float32(target, data_range)
    if pred_f.shape != target_f.shape:
        raise ValueError(f"Shape mismatch: pred {pred_f.shape} vs target {target_f.shape}")

    mse = ((pred_f - target_f) ** 2).mean(dim=(1, 2, 3)).clamp_min(eps)
    psnr = 10.0 * torch.log10((data_range ** 2) / mse)
    return float(psnr.mean().item())


@torch.no_grad()
def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> float:
    """Compute mean batch SSIM with an 11x11 Gaussian window in float32."""
    pred_f = _ensure_4d_float32(pred, data_range)
    target_f = _ensure_4d_float32(target, data_range)
    if pred_f.shape != target_f.shape:
        raise ValueError(f"Shape mismatch: pred {pred_f.shape} vs target {target_f.shape}")

    channels = pred_f.shape[1]
    pad = window_size // 2
    window = _gaussian_window_cpu(window_size, sigma, channels).to(device=pred_f.device)

    mu1 = F.conv2d(pred_f, window, padding=pad, groups=channels)
    mu2 = F.conv2d(target_f, window, padding=pad, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (F.conv2d(pred_f * pred_f, window, padding=pad, groups=channels) - mu1_sq).clamp_min(0.0)
    sigma2_sq = (F.conv2d(target_f * target_f, window, padding=pad, groups=channels) - mu2_sq).clamp_min(0.0)
    sigma12 = F.conv2d(pred_f * target_f, window, padding=pad, groups=channels) - mu1_mu2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / denominator.clamp_min(1.0e-12)
    return float(ssim_map.mean(dim=(1, 2, 3)).mean().item())


@lru_cache(maxsize=4)
def _mexh_cwt_bank_cpu(width: int = 30) -> Tuple[torch.Tensor, int]:
    """
    Build stacked 1D Mexican-hat (Ricker) CWT kernels matching `pyssim` / `xg416/DATUM`
    (`CWSSIM_static.py`: `pywt.cwt(sig, np.arange(1, 31), 'mexh')`).
    """
    x = np.linspace(-8.0, 8.0, 1024)
    step = x[1] - x[0]
    psi = (2.0 / (math.sqrt(3.0) * (math.pi ** 0.25))) * (1.0 - x ** 2) * np.exp(-0.5 * x ** 2)
    int_psi = np.cumsum(psi) * step

    raw_kernels = []
    max_left = 0
    max_right = 0
    for scale in range(1, width + 1):
        j = np.floor(np.arange(scale * 16.0 + 1) / (scale * step)).astype(int)
        if j.max() >= len(int_psi):
            j = np.delete(j, np.where(j >= len(int_psi))[0])
        k = int_psi[j][::-1]
        k_diff = -math.sqrt(scale) * np.diff(np.concatenate([[0.0], k, [0.0]]))
        k_corr = k_diff[::-1].copy()
        orig_m = len(k)
        start = 1 + int(math.floor((orig_m - 2) / 2.0))
        anchor_idx = len(k_corr) - 1 - start
        max_left = max(max_left, anchor_idx)
        max_right = max(max_right, len(k_corr) - 1 - anchor_idx)
        raw_kernels.append((k_corr, anchor_idx))

    center = max(max_left, max_right)
    k_max = 2 * center + 1
    bank = np.zeros((width, 1, k_max), dtype=np.float32)
    for idx, (k_corr, anchor_idx) in enumerate(raw_kernels):
        left = center - anchor_idx
        bank[idx, 0, left : left + len(k_corr)] = k_corr
    return torch.from_numpy(bank), center


@torch.no_grad()
def compute_cw_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    width: int = 30,
    k_const: float = 0.01,
    data_range: float = 1.0,
) -> float:
    """
    Compute CW-SSIM matching `xg416/DATUM` (`pyssim` `cw_ssim_value` with 1D 'mexh' CWT,
    widths=1..30, K=0.01 on [0, 255] grayscale), accelerated on PyTorch/GPU via `F.conv1d`.
    """
    pred_f = _ensure_4d_float32(pred, data_range)
    target_f = _ensure_4d_float32(target, data_range)
    if pred_f.shape != target_f.shape:
        raise ValueError(f"Shape mismatch: pred {pred_f.shape} vs target {target_f.shape}")

    scale_255 = 255.0 / data_range
    if pred_f.shape[1] == 3:
        luma_w = torch.tensor([0.299, 0.587, 0.114], device=pred_f.device, dtype=torch.float32).view(1, 3, 1, 1)
        g_pred = (pred_f * scale_255 * luma_w).sum(dim=1)
        g_target = (target_f * scale_255 * luma_w).sum(dim=1)
    else:
        g_pred = pred_f.mean(dim=1) * scale_255
        g_target = target_f.mean(dim=1) * scale_255

    b, h, w = g_pred.shape
    sig_pair = torch.cat([g_pred, g_target], dim=0).reshape(2 * b, 1, h * w)
    bank_cpu, center = _mexh_cwt_bank_cpu(width)
    bank = bank_cpu.to(device=pred_f.device)

    cwt_both = F.conv1d(sig_pair, bank, padding=center)
    cwt1, cwt2 = cwt_both[:b], cwt_both[b:]

    num1 = 2.0 * (cwt1.abs() * cwt2.abs()).sum(dim=1) + k_const
    den1 = cwt1.square().sum(dim=1) + cwt2.square().sum(dim=1) + k_const

    prod = cwt1 * cwt2
    num2 = 2.0 * prod.sum(dim=1).abs() + k_const
    den2 = 2.0 * prod.abs().sum(dim=1) + k_const

    cw_map = (num1 / den1) * (num2 / den2)
    return float(cw_map.mean().item())


@torch.no_grad()
def evaluate_batch_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> Dict[str, float]:
    """Compute PSNR, SSIM, and DATUM-compatible CW-SSIM for a prediction/target pair."""
    return {
        "psnr": compute_psnr(pred, target, data_range=data_range),
        "ssim": compute_ssim(pred, target, data_range=data_range),
        "cw_ssim": compute_cw_ssim(pred, target, data_range=data_range),
    }
