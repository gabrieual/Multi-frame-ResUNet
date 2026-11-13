"""SOLID Contracts, Interfaces, and Data Transfer Objects for Multi-Frame ResUNet."""

from dataclasses import dataclass, field
from typing import Dict, Protocol, Tuple, runtime_checkable
import torch


@dataclass(frozen=True)
class BurstBatch:
    """Immutable batch DTO produced by StaticBurstDataset."""

    degraded: torch.Tensor  # [B, N, 3, H, W] in [0, 1]
    gt: torch.Tensor        # [B, 3, H, W] in [0, 1]
    sample_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelPrediction:
    """Structured output returned by MultiFrameResUNet."""

    restored: torch.Tensor              # [B, 3, H, W] un-clamped (anchor + residual)
    anchor: torch.Tensor                # [B, 3, H, W] I_anchor used as geometric base
    residual: torch.Tensor              # [B, 3, H, W] predicted residual R
    variance_map: torch.Tensor          # [B, 1, H, W] temporal std-dev map U_var
    stabilized_frames: torch.Tensor     # [B, N, 3, H, W] frames after optional stabilization


@dataclass(frozen=True)
class LossBreakdown:
    """Detailed loss result returned by CompositeLoss."""

    total: torch.Tensor
    components: Dict[str, float] = field(default_factory=dict)


@runtime_checkable
class IStabilizer(Protocol):
    """Interface Segregation: global frame stabilization strategy."""

    def stabilize(self, frames: torch.Tensor) -> torch.Tensor:
        """Aligns [B, N, C, H, W] frames to suppress camera shake."""
        ...


@runtime_checkable
class IAnchorBuilder(Protocol):
    """Interface Segregation: static reference anchor synthesis strategy."""

    def build(self, frames: torch.Tensor) -> torch.Tensor:
        """Synthesizes [B, 3, H, W] anchor frame from [B, N, 3, H, W] burst."""
        ...


@runtime_checkable
class ILossComponent(Protocol):
    """Open/Closed interface for individual spatial/spectral loss terms."""

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes scalar loss between restored [B, 3, H, W] and gt [B, 3, H, W]."""
        ...


@runtime_checkable
class ICheckpointer(Protocol):
    """Dependency Inversion contract for saving and resuming model state."""

    def save(
        self,
        filename: str,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        best_psnr: float,
        config: Dict,
    ) -> None:
        ...

    def load(
        self,
        filename: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        map_location: str | torch.device = "cpu",
    ) -> Dict:
        ...
