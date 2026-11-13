"""Sobel gradient edge preservation loss with replicate border padding."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("edge")
class SobelEdgeLoss(nn.Module):
    """L1 loss on horizontal and vertical Sobel gradients across RGB channels."""

    def __init__(self) -> None:
        super().__init__()
        kx = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ) / 4.0
        ky = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ) / 4.0
        self.register_buffer("sobel_x", kx.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer("sobel_y", ky.view(1, 1, 3, 3).repeat(3, 1, 1, 1))

    def _gradients(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.pad(img.float(), (1, 1, 1, 1), mode="replicate")
        gx = F.conv2d(x, self.sobel_x, padding=0, groups=3)
        gy = F.conv2d(x, self.sobel_y, padding=0, groups=3)
        return gx, gy

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type=pred.device.type, enabled=False):
            gx_p, gy_p = self._gradients(pred)
            gx_t, gy_t = self._gradients(target)
            return F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.compute(pred, target)
