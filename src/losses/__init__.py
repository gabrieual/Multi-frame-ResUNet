"""Spatial and spectral loss components for static burst restoration."""

from src.losses.charbonnier import CharbonnierLoss
from src.losses.composite import CompositeLoss, build_loss
from src.losses.cw_ssim_loss import ComplexWaveletSSIMLoss
from src.losses.edge_loss import SobelEdgeLoss
from src.losses.fft_loss import ComplexFFTLoss
from src.losses.ssim_loss import SSIMLoss

__all__ = [
    "CharbonnierLoss",
    "ComplexFFTLoss",
    "ComplexWaveletSSIMLoss",
    "SobelEdgeLoss",
    "SSIMLoss",
    "CompositeLoss",
    "build_loss",
]
