"""Global 2D-FFT frame stabilization strategies (Liskov-substitutable)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.registry import STABILIZER_REGISTRY


@STABILIZER_REGISTRY.register("none", "noop")
class NoOpStabilizer(nn.Module):
    """Identity stabilizer used during supervised training on aligned synthetic bursts."""

    def stabilize(self, frames: torch.Tensor) -> torch.Tensor:
        return frames

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return frames


@STABILIZER_REGISTRY.register("fft_global")
class FFTGlobalStabilizer(nn.Module):
    """
    Zero-mean centroid global stabilizer using 2D-FFT cross-correlation.
    Removes camera shake while preserving zero-mean atmospheric tilt statistics.
    """

    def __init__(self, max_shift: int = 16) -> None:
        super().__init__()
        self.max_shift = max_shift

    @torch.no_grad()
    def stabilize(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: [B, N, C, H, W] in [0, 1]
        Returns:
            stabilized: [B, N, C, H, W]
        """
        orig_dtype = frames.dtype
        b, n, c, h, w = frames.shape
        if n <= 1:
            return frames

        gray = frames.float().mean(dim=2)  # [B, N, H, W]
        # Subtract spatial mean to focus cross-correlation on edges/structure
        gray_zm = gray - gray.mean(dim=(-2, -1), keepdim=True)
        ref_fft = torch.fft.rfft2(gray_zm[:, 0:1])  # [B, 1, H, W//2+1]
        cur_fft = torch.fft.rfft2(gray_zm)           # [B, N, H, W//2+1]

        xcorr = torch.fft.irfft2(ref_fft * cur_fft.conj(), s=(h, w))  # [B, N, H, W]
        flat_idx = xcorr.view(b, n, -1).argmax(dim=-1)                # [B, N]
        dy = torch.div(flat_idx, w, rounding_mode="floor").float()
        dx = (flat_idx % w).float()

        # Wrap circular indices to signed displacements [-H/2, H/2]
        dy = torch.where(dy > h * 0.5, dy - h, dy).clamp(-self.max_shift, self.max_shift)
        dx = torch.where(dx > w * 0.5, dx - w, dx).clamp(-self.max_shift, self.max_shift)

        # Anchor at zero-mean sequence centroid so mean scene geometry does not drift
        dy = dy - dy.mean(dim=1, keepdim=True)
        dx = dx - dx.mean(dim=1, keepdim=True)

        # Shift using bilinear grid_sample in float32 with border replication
        yy = torch.arange(h, device=frames.device, dtype=torch.float32)
        xx = torch.arange(w, device=frames.device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")  # [H, W]

        shift_y = grid_y.view(1, 1, h, w) - dy.view(b, n, 1, 1)
        shift_x = grid_x.view(1, 1, h, w) - dx.view(b, n, 1, 1)

        norm_y = 2.0 * shift_y / max(h - 1, 1) - 1.0
        norm_x = 2.0 * shift_x / max(w - 1, 1) - 1.0
        grid = torch.stack([norm_x, norm_y], dim=-1).view(b * n, h, w, 2)

        warped = F.grid_sample(
            frames.float().view(b * n, c, h, w),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return warped.view(b, n, c, h, w).to(orig_dtype)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.stabilize(frames)
