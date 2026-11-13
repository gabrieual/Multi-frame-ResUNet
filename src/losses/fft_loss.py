"""2D Fourier spectral loss computed in forced float32."""

import torch
import torch.nn as nn

from src.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("fft")
class ComplexFFTLoss(nn.Module):
    """L1 distance between real and imaginary parts of 2D Real FFT (norm='ortho')."""

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type=pred.device.type, enabled=False):
            pred_fft = torch.fft.rfft2(pred.float(), norm="ortho")
            target_fft = torch.fft.rfft2(target.float(), norm="ortho")
            loss_real = torch.abs(pred_fft.real - target_fft.real).mean()
            loss_imag = torch.abs(pred_fft.imag - target_fft.imag).mean()
            return loss_real + loss_imag

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.compute(pred, target)
