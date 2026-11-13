"""Spatial and spectral loss components for static burst restoration."""

from src.losses.charbonnier import CharbonnierLoss
from src.losses.composite import CompositeLoss, build_loss
from src.losses.edge_loss import SobelEdgeLoss
from src.losses.fft_loss import ComplexFFTLoss

__all__ = [
    "CharbonnierLoss",
    "ComplexFFTLoss",
    "SobelEdgeLoss",
    "CompositeLoss",
    "build_loss",
]
