"""Static reference anchor builders (Mean vs. CDSP Frequency-Aware Reference Frame)."""

import torch
import torch.nn as nn

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
