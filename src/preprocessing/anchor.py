"""Static reference anchor builders (Mean, CDSP FRF, and Lucky Sharpness Fusion)."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.registry import ANCHOR_REGISTRY


@ANCHOR_REGISTRY.register("mean")
class MeanAnchorBuilder(nn.Module):
    """Computes temporal mean I_mean = (1/N) * sum(I_t) as geometric anchor."""

    @torch.no_grad()
    def build(self, frames: torch.Tensor) -> torch.Tensor:
        orig_dtype = frames.dtype
        return frames.float().mean(dim=1).to(orig_dtype)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.build(frames)


@ANCHOR_REGISTRY.register("frf")
class FRFAnchorBuilder(nn.Module):
    """
    Frequency-Aware Reference Frame (CDSP): weights each pixel temporally by how
    frequently its intensity recurs across the burst (Parzen-window mode estimation).
    """

    def __init__(self, bandwidth_h: float = 0.05, sigma: float = 0.5) -> None:
        super().__init__()
        self.bandwidth_h = float(bandwidth_h)
        self.sigma = float(sigma)

    @torch.no_grad()
    def build(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: [B, N, 3, H, W] in [0, 1]
        Returns:
            frf: [B, 3, H, W]
        """
        orig_dtype = frames.dtype
        frames_f32 = frames.float()
        b, n, c, h, w = frames_f32.shape
        if n <= 1:
            return frames[:, 0]

        denom = 2.0 * (self.bandwidth_h ** 2) + 1e-8
        # Loop over n to avoid materializing an [B, N, N, C, H, W] tensor in VRAM
        n_l_list = []
        for t in range(n):
            diff_sq_t = (frames_f32 - frames_f32[:, t : t + 1]).square().sum(dim=2)
            n_l_t = torch.exp(-diff_sq_t / denom).sum(dim=1)  # [B, H, W]
            n_l_list.append(n_l_t)

        n_l = torch.stack(n_l_list, dim=1)                    # [B, N, H, W]
        weights = torch.softmax(self.sigma * n_l, dim=1)      # [B, N, H, W]
        frf = (frames_f32 * weights.unsqueeze(2)).sum(dim=1)  # [B, C, H, W]
        return frf.to(orig_dtype)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.build(frames)


@ANCHOR_REGISTRY.register("lucky")
class LuckyAnchorBuilder(nn.Module):
    """
    Sharpness- and similarity-weighted anchor inspired by CLEAR & CVPR UG2+ winners:
      1. Computes an outlier-resistant reference I_ref via FRF.
      2. Scores each frame locally by Sobel gradient energy smoothed over 7x7 regions.
      3. Suppresses geometrically shifted outliers via CLEAR similarity 1 / (lambda_s + ||I_t - I_ref||^2).
      4. Fuses frames temporally with contrast-normalized Softmax weights.
    """

    def __init__(
        self,
        bandwidth_h: float = 0.05,
        sigma: float = 0.5,
        tau: float = 4.0,
        lambda_s: float = 0.02,
        pool_kernel: int = 7,
    ) -> None:
        super().__init__()
        self.frf_builder = FRFAnchorBuilder(bandwidth_h=bandwidth_h, sigma=sigma)
        self.tau = float(tau)
        self.lambda_s = float(lambda_s)
        self.pool_kernel = int(pool_kernel)
        if self.pool_kernel < 1 or self.pool_kernel % 2 == 0:
            raise ValueError(f"pool_kernel must be a positive odd integer, got {pool_kernel}")

        kx = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ) / 4.0
        ky = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ) / 4.0
        self.register_buffer("sobel_x", kx.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y", ky.view(1, 1, 3, 3), persistent=False)

    @torch.no_grad()
    def build(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: [B, N, 3, H, W] in [0, 1]
        Returns:
            lucky_anchor: [B, 3, H, W]
        """
        orig_dtype = frames.dtype
        b, n, _, h, w = frames.shape
        if n <= 1:
            return frames[:, 0]

        with torch.amp.autocast(device_type=frames.device.type, enabled=False):
            frames_f32 = frames.float()
            ref = self.frf_builder.build(frames_f32)  # [B, 3, H, W]

            # Luminance for local gradient sharpness [B*N, 1, H, W]
            luma = (
                0.299 * frames_f32[:, :, 0:1]
                + 0.587 * frames_f32[:, :, 1:2]
                + 0.114 * frames_f32[:, :, 2:3]
            ).reshape(b * n, 1, h, w)

            luma_pad = F.pad(luma, (1, 1, 1, 1), mode="replicate")
            gx = F.conv2d(luma_pad, self.sobel_x.to(device=luma.device, dtype=torch.float32))
            gy = F.conv2d(luma_pad, self.sobel_y.to(device=luma.device, dtype=torch.float32))
            grad_mag = torch.sqrt(gx * gx + gy * gy + 1.0e-8)

            pad_k = self.pool_kernel // 2
            local_sharpness = F.avg_pool2d(
                grad_mag,
                kernel_size=self.pool_kernel,
                stride=1,
                padding=pad_k,
                count_include_pad=False,
            ).reshape(b, n, h, w)

            # CLEAR anti-outlier similarity against FRF reference
            diff_sq = (frames_f32 - ref.unsqueeze(1)).square().mean(dim=2)  # [B, N, H, W]
            similarity = 1.0 / (self.lambda_s + diff_sq)

            priority = local_sharpness * similarity
            norm_priority = priority / priority.mean(dim=1, keepdim=True).clamp_min(1.0e-6)
            weights = torch.softmax(self.tau * norm_priority, dim=1)  # [B, N, H, W]

            lucky = (frames_f32 * weights.unsqueeze(2)).sum(dim=1)
        return lucky.to(orig_dtype)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.build(frames)


def build_anchor_builder(
    model_cfg: Dict[str, Any],
    anchor_override: Optional[str] = None,
) -> nn.Module:
    """Instantiate the configured anchor builder cleanly from `model` config."""
    anchor_type = str(anchor_override or model_cfg.get("anchor_type", "mean")).strip().lower()
    if anchor_type == "frf":
        return ANCHOR_REGISTRY.build(
            "frf",
            bandwidth_h=float(model_cfg.get("frf_bandwidth_h", 0.05)),
            sigma=float(model_cfg.get("frf_sigma", 0.5)),
        )
    if anchor_type == "lucky":
        return ANCHOR_REGISTRY.build(
            "lucky",
            bandwidth_h=float(model_cfg.get("frf_bandwidth_h", 0.05)),
            sigma=float(model_cfg.get("frf_sigma", 0.5)),
            tau=float(model_cfg.get("lucky_tau", 4.0)),
            lambda_s=float(model_cfg.get("lucky_lambda_s", 0.02)),
        )
    return ANCHOR_REGISTRY.build(anchor_type)
