"""Stage-1 lightweight pre-stem sub-pixel geometric aligner for static burst frames."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FlowConvBlock(nn.Module):
    """Lightweight residual-free conv block with SiLU for smooth flow estimation."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PreStemFlowAligner(nn.Module):
    """
    Stage-1 pre-stem geometric turbulence aligner (two-stage paradigm inspired by
    Anantrasirichai PRL 2023 & MAMAT AVSS 2025).

    Predicts a bounded dense sub-pixel displacement field per burst frame relative to
    an outlier-resistant reference frame (I_FRF) and warps all frames jointly before
    the main Multi-Frame ResUNet refinement stage. Zero-initialized at the flow head
    so step-0 output starts identically at unwarped frames.
    """

    def __init__(self, hidden_ch: int = 24, max_shift: float = 12.0) -> None:
        super().__init__()
        ch = int(hidden_ch)
        self.max_shift = float(max_shift)

        self.enc1 = _FlowConvBlock(6, ch)
        self.down = nn.Conv2d(ch, ch * 2, kernel_size=2, stride=2)
        self.enc2 = _FlowConvBlock(ch * 2, ch * 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = _FlowConvBlock(ch * 3, ch)

        self.flow_head = nn.Conv2d(ch, 2, kernel_size=3, padding=1)
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)

    @staticmethod
    def _pad_to_even(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        _, _, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h == 0 and pad_w == 0:
            return x, h, w
        mode = "reflect" if (h > 1 and w > 1) else "replicate"
        return F.pad(x, (0, pad_w, 0, pad_h), mode=mode), h, w

    def _estimate_flow_pixels(self, pairs: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        # Operate at 1/2 spatial scale for low VRAM overhead with N=16 bursts and smooth flow fields.
        low = F.avg_pool2d(pairs, kernel_size=2, stride=2) if min(target_h, target_w) >= 8 else pairs
        low_pad, low_h, low_w = self._pad_to_even(low)

        f1 = self.enc1(low_pad)
        f2 = self.enc2(self.down(f1))
        up = self.up(f2)
        if up.shape[-2:] != f1.shape[-2:]:
            up = F.interpolate(up, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        dec = self.dec1(torch.cat([up, f1], dim=1))[:, :, :low_h, :low_w]

        raw_flow = torch.tanh(self.flow_head(dec)) * self.max_shift
        if raw_flow.shape[-2:] != (target_h, target_w):
            return F.interpolate(raw_flow, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return raw_flow

    def forward(
        self, frames: torch.Tensor, reference: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            frames: [B, N, 3, H, W] in [0, 1]
            reference: [B, 3, H, W] distortion-averaged reference (e.g. I_FRF)
        Returns:
            aligned_frames: [B, N, 3, H, W] aligned to `reference`
            flow_px: [B, N, 2, H, W] estimated pixel displacements `(dx, dy)`
        """
        b, n, c, h, w = frames.shape
        ref_exp = reference.unsqueeze(1).expand(b, n, c, h, w)
        pairs = torch.cat([frames, ref_exp], dim=2).reshape(b * n, c * 2, h, w)
        flow_px = self._estimate_flow_pixels(pairs, target_h=h, target_w=w)

        orig_dtype = frames.dtype
        with torch.amp.autocast(device_type=frames.device.type, enabled=False):
            frames_f32 = frames.reshape(b * n, c, h, w).float()
            flow_f32 = flow_px.float()

            ys = torch.linspace(-1.0, 1.0, h, device=frames.device, dtype=torch.float32)
            xs = torch.linspace(-1.0, 1.0, w, device=frames.device, dtype=torch.float32)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]

            norm_x = 2.0 / float(max(w - 1, 1))
            norm_y = 2.0 / float(max(h - 1, 1))
            delta_x = flow_f32[:, 0] * norm_x
            delta_y = flow_f32[:, 1] * norm_y
            grid = base_grid + torch.stack([delta_x, delta_y], dim=-1)

            warped = F.grid_sample(
                frames_f32,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )

        return warped.reshape(b, n, c, h, w).to(orig_dtype), flow_px.reshape(b, n, 2, h, w)
