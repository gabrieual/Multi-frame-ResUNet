"""Differentiable Charbonnier L1 loss (float32)."""

import torch
import torch.nn as nn

from src.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("charbonnier")
class CharbonnierLoss(nn.Module):
    """L_char(x, y) = mean(sqrt(||x - y||^2 + eps^2))."""

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps_sq = float(eps) ** 2

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type=pred.device.type, enabled=False):
            diff = pred.float() - target.float()
            return torch.mean(torch.sqrt(diff * diff + self.eps_sq))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.compute(pred, target)
