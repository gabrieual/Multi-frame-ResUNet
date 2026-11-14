"""Open/Closed Composite Loss orchestrating weighted ILossComponent terms."""

from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn

from src.contracts import LossBreakdown
from src.losses.charbonnier import CharbonnierLoss
from src.losses.edge_loss import SobelEdgeLoss
from src.losses.fft_loss import ComplexFFTLoss
from src.losses.ssim_loss import SSIMLoss


class CompositeLoss(nn.Module):
    """Aggregates named loss components and returns total tensor + scalar breakdown."""

    def __init__(self, terms: List[Tuple[str, float, nn.Module]]) -> None:
        super().__init__()
        self.weights: Dict[str, float] = {}
        self.modules_map = nn.ModuleDict()
        for name, weight, module in terms:
            if weight > 0.0:
                self.weights[name] = float(weight)
                self.modules_map[name] = module

        if not self.modules_map:
            raise ValueError("CompositeLoss requires at least one loss term with weight > 0.")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> LossBreakdown:
        total = pred.new_zeros((), dtype=torch.float32)
        breakdown: Dict[str, float] = {}
        for name, module in self.modules_map.items():
            raw_val = module(pred, target)
            w = self.weights[name]
            total = total + w * raw_val.float()
            breakdown[name] = float(raw_val.detach().item())
        breakdown["total"] = float(total.detach().item())
        return LossBreakdown(total=total, components=breakdown)


def build_loss(cfg: Dict[str, Any]) -> CompositeLoss:
    loss_cfg = cfg["loss"]
    terms: List[Tuple[str, float, nn.Module]] = [
        (
            "charbonnier",
            float(loss_cfg.get("charbonnier_weight", 1.0)),
            CharbonnierLoss(eps=float(loss_cfg.get("charbonnier_eps", 1e-3))),
        ),
        (
            "fft",
            float(loss_cfg.get("fft_weight", 0.05)),
            ComplexFFTLoss(),
        ),
        (
            "edge",
            float(loss_cfg.get("edge_weight", 0.0)),
            SobelEdgeLoss(),
        ),
        (
            "ssim",
            float(loss_cfg.get("ssim_weight", 0.0)),
            SSIMLoss(),
        ),
    ]
    return CompositeLoss(terms)
