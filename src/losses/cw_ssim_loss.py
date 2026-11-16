"""Multi-scale Complex Wavelet SSIM (CW-SSIM) loss in forced float32."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.registry import LOSS_REGISTRY


def _build_complex_wavelet_bank(
    num_orientations: int = 6,
    kernel_size: int = 7,
    sigma: float = 1.6,
    wavelength: float = 3.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build 6-orientation complex quadrature Gabor wavelet filters (real & imaginary)."""
    half = kernel_size // 2
    coords = torch.arange(-half, half + 1, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    gauss = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma * sigma))

    real_list, imag_list = [], []
    for k in range(num_orientations):
        theta = k * math.pi / num_orientations
        x_theta = xx * math.cos(theta) + yy * math.sin(theta)
        phase = 2.0 * math.pi * x_theta / wavelength
        real_k = gauss * torch.cos(phase)
        imag_k = gauss * torch.sin(phase)
        real_k = real_k - real_k.mean()
        imag_k = imag_k - imag_k.mean()
        norm = torch.sqrt(real_k.square().sum() + imag_k.square().sum()).clamp_min(1.0e-6)
        real_list.append(real_k / norm)
        imag_list.append(imag_k / norm)

    real_bank = torch.stack(real_list, dim=0).unsqueeze(1).contiguous()  # [O, 1, K, K]
    imag_bank = torch.stack(imag_list, dim=0).unsqueeze(1).contiguous()  # [O, 1, K, K]
    return real_bank, imag_bank


@LOSS_REGISTRY.register("cw_ssim")
class ComplexWaveletSSIMLoss(nn.Module):
    """
    Differentiable `1.0 - CW-SSIM(pred, target)` across dyadic scales and 6 orientations.
    Insensitive to small sub-pixel geometric shifts while strongly penalizing local phase distortion
    (Sampat et al., IEEE TIP 2009; Anantrasirichai et al., CLEAR IEEE TIP 2013).
    """

    def __init__(
        self,
        num_scales: int = 3,
        num_orientations: int = 6,
        window_size: int = 7,
        kernel_size: int = 7,
        k_const: float = 1.0e-4,
    ) -> None:
        super().__init__()
        self.num_scales = int(num_scales)
        self.num_orientations = int(num_orientations)
        self.window_size = int(window_size)
        self.kernel_size = int(kernel_size)
        self.k_const = float(k_const)

        real_bank, imag_bank = _build_complex_wavelet_bank(
            num_orientations=self.num_orientations,
            kernel_size=self.kernel_size,
        )
        self.register_buffer("real_bank", real_bank, persistent=False)
        self.register_buffer("imag_bank", imag_bank, persistent=False)
        self.register_buffer(
            "luma_weights",
            torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def _to_luma(self, img: torch.Tensor) -> torch.Tensor:
        if img.shape[1] == 3:
            w = self.luma_weights.to(device=img.device, dtype=torch.float32)
            return (img * w).sum(dim=1, keepdim=True)
        return img.mean(dim=1, keepdim=True)

    def _cw_ssim_single_scale(
        self,
        x_luma: torch.Tensor,
        y_luma: torch.Tensor,
        real_bank: torch.Tensor,
        imag_bank: torch.Tensor,
    ) -> torch.Tensor:
        pad = self.kernel_size // 2
        pad_mode = "reflect" if (x_luma.shape[-2] > pad and x_luma.shape[-1] > pad) else "replicate"
        x_pad = F.pad(x_luma, (pad, pad, pad, pad), mode=pad_mode)
        y_pad = F.pad(y_luma, (pad, pad, pad, pad), mode=pad_mode)

        cx_r = F.conv2d(x_pad, real_bank)
        cx_i = F.conv2d(x_pad, imag_bank)
        cy_r = F.conv2d(y_pad, real_bank)
        cy_i = F.conv2d(y_pad, imag_bank)

        prod_r = cx_r * cy_r + cx_i * cy_i
        prod_i = cx_i * cy_r - cx_r * cy_i
        mag_x_sq = cx_r.square() + cx_i.square()
        mag_y_sq = cy_r.square() + cy_i.square()

        p_w = self.window_size // 2
        sum_prod_r = F.avg_pool2d(prod_r, kernel_size=self.window_size, stride=1, padding=p_w)
        sum_prod_i = F.avg_pool2d(prod_i, kernel_size=self.window_size, stride=1, padding=p_w)
        sum_mag_x = F.avg_pool2d(mag_x_sq, kernel_size=self.window_size, stride=1, padding=p_w)
        sum_mag_y = F.avg_pool2d(mag_y_sq, kernel_size=self.window_size, stride=1, padding=p_w)

        cross_abs = torch.sqrt(sum_prod_r.square() + sum_prod_i.square() + 1.0e-12)
        cw_map = (2.0 * cross_abs + self.k_const) / (sum_mag_x + sum_mag_y + self.k_const)
        return cw_map.mean()

    def compute_similarity(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute mean multi-scale CW-SSIM similarity in [0, 1] inside forced float32."""
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}")

        with torch.amp.autocast(device_type=pred.device.type, enabled=False):
            x_luma = self._to_luma(pred.float())
            y_luma = self._to_luma(target.float())
            real_bank = self.real_bank.to(device=x_luma.device, dtype=torch.float32)
            imag_bank = self.imag_bank.to(device=x_luma.device, dtype=torch.float32)

            scores = []
            for scale_idx in range(self.num_scales):
                if scale_idx > 0 and min(x_luma.shape[-2], x_luma.shape[-1]) < 16:
                    break
                scores.append(self._cw_ssim_single_scale(x_luma, y_luma, real_bank, imag_bank))
                if scale_idx < self.num_scales - 1:
                    x_luma = F.avg_pool2d(x_luma, kernel_size=2, stride=2)
                    y_luma = F.avg_pool2d(y_luma, kernel_size=2, stride=2)

            return torch.stack(scores).mean()

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - self.compute_similarity(pred, target)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.compute(pred, target)
