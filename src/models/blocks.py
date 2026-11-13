"""Core NAFNet building blocks with fp16-safe LayerNorm2d for T4 GPUs."""

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """
    Channel-first 2D LayerNorm for [B, C, H, W] tensors.
    Always computes statistics in float32 to prevent float16 overflow on T4 GPUs.
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f32 = x.float()
        mean = x_f32.mean(dim=1, keepdim=True)
        var = (x_f32 - mean).square().mean(dim=1, keepdim=True)
        x_norm = (x_f32 - mean) * torch.rsqrt(var + self.eps)
        out = x_norm * self.weight.float() + self.bias.float()
        return out.to(orig_dtype)


class SimpleGate(nn.Module):
    """Splits channels in half and multiplies element-wise (Chen et al., ECCV 2022)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """
    Nonlinear Activation Free Block (NAFNet, ECCV 2022):
      - LayerNorm2d (fp32-safe) + Pointwise + Depthwise 3x3 + SimpleGate
      - Simplified Channel Attention (SCA)
      - Residual FFN branch with learnable scalars beta and gamma initialized to zero.
    """

    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        drop_out_rate: float = 0.0,
    ) -> None:
        super().__init__()
        dw_ch = channels * dw_expand
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_ch, kernel_size=1)
        self.conv2 = nn.Conv2d(dw_ch, dw_ch, kernel_size=3, padding=1, groups=dw_ch)
        self.sg1 = SimpleGate()

        sca_ch = dw_ch // 2
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(sca_ch, sca_ch, kernel_size=1),
        )
        self.conv3 = nn.Conv2d(sca_ch, channels, kernel_size=1)

        ffn_ch = channels * ffn_expand
        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_ch, kernel_size=1)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_ch // 2, channels, kernel_size=1)

        self.dropout1 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )
        self.dropout2 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg1(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg2(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma
