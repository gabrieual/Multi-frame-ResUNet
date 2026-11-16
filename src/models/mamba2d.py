"""Pure-PyTorch 2D Selective State-Space (Mamba2D / SS2D) bottleneck block for global context."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.blocks import LayerNorm2d, SimpleGate


def parallel_selective_scan(a: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    Exact parallel associative prefix scan solving `h_t = a_t * h_{t-1} + u_t` (`h_{-1} = 0`)
    along the last dimension `L` via Hillis-Steele doubling in `O(log L)` vectorized steps.

    Zero in-place ops (autograd-safe) and zero custom CUDA extensions required.
    """
    length = a.shape[-1]
    if length <= 1:
        return u

    cur_a = a
    cur_u = u
    step = 1
    while step < length:
        a_prev = cur_a[..., :-step]
        u_prev = cur_u[..., :-step]
        a_cur = cur_a[..., step:]
        u_cur = cur_u[..., step:]

        next_u_tail = a_cur * u_prev + u_cur
        cur_u = torch.cat([cur_u[..., :step], next_u_tail], dim=-1)

        if step * 2 < length:
            next_a_tail = a_cur * a_prev
            cur_a = torch.cat([cur_a[..., :step], next_a_tail], dim=-1)
        step *= 2

    return cur_u


class BidirectionalSelectiveScan1D(nn.Module):
    """
    Input-dependent bidirectional Mamba S6 scan along 1D sequences `[S, D, L]`.
    Runs forward and backward directions simultaneously in forced float32.
    """

    def __init__(
        self,
        channels: int,
        d_state: int = 8,
        dt_rank: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.d_state = int(d_state)
        self.dt_rank = int(dt_rank) if dt_rank is not None else max(8, math.ceil(self.channels / 16))

        proj_dim = self.dt_rank + 2 * self.d_state
        self.x_proj = nn.Conv1d(self.channels, proj_dim, kernel_size=1, bias=False)
        self.dt_proj = nn.Conv1d(self.dt_rank, self.channels, kernel_size=1, bias=True)

        # Initialize dt_proj bias so softplus(dt_bias) lies in [0.03, 0.3] (tuned for 16..64 bottleneck length)
        with torch.no_grad():
            dt_init = torch.exp(
                torch.linspace(math.log(0.03), math.log(0.3), self.channels)
            ).clamp_min(1.0e-4)
            inv_softplus = dt_init + torch.log(-torch.expm1(-dt_init))
            self.dt_proj.bias.copy_(inv_softplus)

        # S4D / Mamba diagonal log-decay initialization: A = -exp(a_log) < 0
        a_init = 0.25 * torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.channels, 1)
        self.a_log = nn.Parameter(torch.log(a_init))
        self.d_skip = nn.Parameter(torch.full((self.channels,), 0.2, dtype=torch.float32))

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: `[S, D, L]` real tensor
        Returns:
            y_seq: `[S, D, L]` matching `x_seq.dtype`
        """
        orig_dtype = x_seq.dtype
        s, d, length = x_seq.shape

        x_bi = torch.cat([x_seq, x_seq.flip(dims=(-1,))], dim=0)  # [2S, D, L]
        params = self.x_proj(x_bi)
        dt_raw, b_mat, c_mat = torch.split(
            params, [self.dt_rank, self.d_state, self.d_state], dim=1
        )
        dt_raw = self.dt_proj(dt_raw)

        with torch.amp.autocast(device_type=x_seq.device.type, enabled=False):
            x_f32 = x_bi.float()
            dt = F.softplus(dt_raw.float())  # [2S, D, L] > 0
            b_f32 = b_mat.float()            # [2S, N, L]
            c_f32 = c_mat.float()            # [2S, N, L]

            neg_a = -torch.exp(self.a_log.float())  # [D, N] < 0
            log_a = (dt.unsqueeze(2) * neg_a.unsqueeze(0).unsqueeze(-1)).clamp(min=-20.0, max=0.0)
            a_discrete = torch.exp(log_a)           # [2S, D, N, L] in (0, 1]
            u_drive = (dt * x_f32).unsqueeze(2) * b_f32.unsqueeze(1)  # [2S, D, N, L]

            states = parallel_selective_scan(a_discrete, u_drive)     # [2S, D, N, L]
            y_bi = (states * c_f32.unsqueeze(1)).sum(dim=2)           # [2S, D, L]

            y_fwd, y_bwd = y_bi[:s], y_bi[s:].flip(dims=(-1,))
            y_scan = 0.5 * (y_fwd + y_bwd)
            # Balance SSM scan magnitude against d_skip across arbitrary channel widths D
            y_scan = y_scan * torch.rsqrt(y_scan.square().mean(dim=1, keepdim=True) + 1.0e-6)
            y_merged = y_scan + self.d_skip.float().view(1, d, 1) * x_seq.float()

        return y_merged.to(orig_dtype)


class SS2DBlock(nn.Module):
    """
    2D Residual State-Space Block combining separable Row + Col bidirectional
    selective scans (full 2D global receptive field per block in O(H*W * log(max(H,W))))
    with a NAFNet-style SimpleGate FFN branch and zero-init residual scalars.
    """

    def __init__(
        self,
        channels: int,
        d_state: int = 8,
        ssm_expand: float = 0.5,
        ffn_expand: int = 2,
        drop_out_rate: float = 0.0,
    ) -> None:
        super().__init__()
        d_inner = max(16, int(round(channels * ssm_expand)))

        self.norm1 = LayerNorm2d(channels)
        self.in_proj = nn.Conv2d(channels, d_inner * 2, kernel_size=1)
        self.dw_conv = nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner)
        self.act = nn.SiLU(inplace=True)

        self.row_ssm = BidirectionalSelectiveScan1D(d_inner, d_state=d_state)
        self.col_ssm = BidirectionalSelectiveScan1D(d_inner, d_state=d_state)
        self.out_norm = LayerNorm2d(d_inner)
        self.out_proj = nn.Conv2d(d_inner, channels, kernel_size=1)

        ffn_ch = channels * int(ffn_expand)
        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_ch, kernel_size=1)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_ch // 2, channels, kernel_size=1)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        b, _, h, w = inp.shape
        x_in, z_gate = self.in_proj(self.norm1(inp)).chunk(2, dim=1)
        x = self.act(self.dw_conv(x_in))
        d = x.shape[1]

        # Row bidirectional scan: [B*H, D, W] -> [B, D, H, W]
        x_row = self.row_ssm(x.permute(0, 2, 1, 3).reshape(b * h, d, w))
        x = x_row.reshape(b, h, d, w).permute(0, 2, 1, 3)

        # Col bidirectional scan sequentially -> full 2D receptive field in 1 block
        x_col = self.col_ssm(x.permute(0, 3, 1, 2).reshape(b * w, d, h))
        x = x_col.reshape(b, w, d, h).permute(0, 2, 3, 1).contiguous()

        x = self.out_norm(x) * F.silu(z_gate)
        x = self.dropout1(self.out_proj(x))
        y = inp + x * self.beta

        f = self.norm2(y)
        f = self.conv5(self.sg2(self.conv4(f)))
        f = self.dropout2(f)
        return y + f * self.gamma
