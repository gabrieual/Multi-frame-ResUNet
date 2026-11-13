"""Overlap-tile full-resolution inference with fixed 256x256 SCA-consistent tiles."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..contracts import ModelPrediction
from ..models.resunet import MultiFrameResUNet


def _compute_tile_starts(length: int, tile_size: int, stride: int) -> List[int]:
    """Compute tile start coordinates covering `[0, length)` cleanly."""
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def _pad_burst_bottom_right(
    frames: torch.Tensor, min_h: int, min_w: int
) -> Tuple[torch.Tensor, int, int]:
    """
    Pad `[B, N, 3, H, W]` on bottom/right so spatial dimensions are at least `(min_h, min_w)`.
    Flattens to 4D before `F.pad` because PyTorch `reflect` padding does not support 5D tensors.
    """
    b, n, c, h, w = frames.shape
    pad_h = max(0, min_h - h)
    pad_w = max(0, min_w - w)
    if pad_h == 0 and pad_w == 0:
        return frames, h, w

    mode = "reflect" if (h > pad_h and w > pad_w) else "replicate"
    flat = frames.reshape(b * n, c, h, w)
    padded = F.pad(flat, (0, pad_w, 0, pad_h), mode=mode)
    return padded.reshape(b, n, c, h + pad_h, w + pad_w), h, w


class TiledInference:
    """
    Overlap-tile evaluator preserving training `AdaptiveAvgPool2d(1)` SCA statistics
    by slicing arbitrary resolutions into fixed `tile_size=256` tiles with Hann blending.
    """

    def __init__(
        self,
        tile_size: int = 256,
        overlap: int = 64,
        max_eval_windows: int = 4,
        stabilize_in_eval: bool = False,
    ) -> None:
        if tile_size <= 0:
            raise ValueError(f"tile_size must be positive, got {tile_size}")
        if overlap < 0 or overlap >= tile_size:
            raise ValueError(f"Require 0 <= overlap < tile_size, got overlap={overlap}, tile={tile_size}")
        if max_eval_windows <= 0:
            raise ValueError(f"max_eval_windows must be positive, got {max_eval_windows}")

        self.tile_size = int(tile_size)
        self.overlap = int(overlap)
        self.stride = self.tile_size - self.overlap
        self.max_eval_windows = int(max_eval_windows)
        self.stabilize_in_eval = bool(stabilize_in_eval)

        win_1d = torch.hann_window(self.tile_size, periodic=False, dtype=torch.float32).clamp_min(1.0e-3)
        self._base_window = torch.outer(win_1d, win_1d).unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]

    @torch.inference_mode()
    def predict_burst(
        self,
        model: MultiFrameResUNet,
        frames: torch.Tensor,
        amp_dtype: Optional[torch.dtype] = None,
    ) -> ModelPrediction:
        """
        Run overlap-tile inference on a `[B, N, 3, H, W]` burst.

        Global stabilization (when enabled) executes once on full resolution BEFORE tiling
        so adjacent tiles share identical sub-pixel frame registration.
        """
        if frames.ndim != 5:
            raise ValueError(f"Expected 5D [B, N, 3, H, W] burst, got shape {tuple(frames.shape)}")

        device = frames.device
        b, n, _, orig_h, orig_w = frames.shape

        # Run full-frame stabilization once before slicing tiles
        work_frames = model.stabilizer.stabilize(frames) if self.stabilize_in_eval else frames
        padded_frames, _, _ = _pad_burst_bottom_right(work_frames, self.tile_size, self.tile_size)
        _, _, _, pad_h, pad_w = padded_frames.shape

        ys = _compute_tile_starts(pad_h, self.tile_size, self.stride)
        xs = _compute_tile_starts(pad_w, self.tile_size, self.stride)
        window = self._base_window.to(device=device, dtype=torch.float32)

        restored_acc = torch.zeros((b, 3, pad_h, pad_w), device=device, dtype=torch.float32)
        anchor_acc = torch.zeros((b, 3, pad_h, pad_w), device=device, dtype=torch.float32)
        residual_acc = torch.zeros((b, 3, pad_h, pad_w), device=device, dtype=torch.float32)
        var_acc = torch.zeros((b, 1, pad_h, pad_w), device=device, dtype=torch.float32)
        weight_acc = torch.zeros((b, 1, pad_h, pad_w), device=device, dtype=torch.float32)

        use_autocast = amp_dtype is not None and amp_dtype != torch.float32 and device.type == "cuda"
        effective_dtype = amp_dtype if amp_dtype is not None else torch.float32
        if use_autocast and effective_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            effective_dtype = torch.float16

        for y in ys:
            for x in xs:
                tile = padded_frames[:, :, :, y : y + self.tile_size, x : x + self.tile_size]
                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=effective_dtype,
                    enabled=use_autocast,
                ):
                    pred = model(tile, apply_stabilizer=False)

                restored_acc[:, :, y : y + self.tile_size, x : x + self.tile_size] += (
                    pred.restored.float() * window
                )
                anchor_acc[:, :, y : y + self.tile_size, x : x + self.tile_size] += (
                    pred.anchor.float() * window
                )
                residual_acc[:, :, y : y + self.tile_size, x : x + self.tile_size] += (
                    pred.residual.float() * window
                )
                var_acc[:, :, y : y + self.tile_size, x : x + self.tile_size] += (
                    pred.variance_map.float() * window
                )
                weight_acc[:, :, y : y + self.tile_size, x : x + self.tile_size] += window

        denom = weight_acc.clamp_min(1.0e-6)
        return ModelPrediction(
            restored=(restored_acc / denom)[:, :, :orig_h, :orig_w],
            anchor=(anchor_acc / denom)[:, :, :orig_h, :orig_w],
            residual=(residual_acc / denom)[:, :, :orig_h, :orig_w],
            variance_map=(var_acc / denom)[:, :, :orig_h, :orig_w],
            stabilized_frames=work_frames,
        )

    def select_eval_windows(self, total_frames: int, num_frames: int) -> List[List[int]]:
        """
        Partition a `T`-frame sequence into up to `max_eval_windows` evenly distributed
        `num_frames`-length evaluation bursts matching training sampling span.
        """
        if total_frames <= 0:
            raise ValueError(f"Sequence has {total_frames} frames.")
        if total_frames <= num_frames:
            return [np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()]

        num_windows = min(self.max_eval_windows, total_frames - num_frames + 1)
        if num_windows == 1:
            return [np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()]

        max_offset = num_windows - 1
        windows: List[List[int]] = []
        for w in range(num_windows):
            end_idx = total_frames - 1 - (max_offset - w)
            indices = np.linspace(w, end_idx, num_frames, dtype=int).tolist()
            windows.append(indices)
        return windows

    @torch.inference_mode()
    def predict_sequence(
        self,
        model: MultiFrameResUNet,
        sequence: torch.Tensor,
        num_frames: int,
        amp_dtype: Optional[torch.dtype] = None,
    ) -> ModelPrediction:
        """
        Evaluate a full temporal sequence `[T, 3, H, W]` (or `[1, T, 3, H, W]`)
        by averaging overlap-tile predictions over multiple temporal windows.
        """
        if sequence.ndim == 5:
            if sequence.shape[0] != 1:
                raise ValueError("predict_sequence expects batch size 1 or 4D [T, 3, H, W]")
            sequence = sequence.squeeze(0)
        if sequence.ndim != 4:
            raise ValueError(f"Expected 4D [T, 3, H, W] sequence, got {tuple(sequence.shape)}")

        total_frames = int(sequence.shape[0])
        windows = self.select_eval_windows(total_frames, num_frames)

        restored_sum: Optional[torch.Tensor] = None
        anchor_sum: Optional[torch.Tensor] = None
        residual_sum: Optional[torch.Tensor] = None
        var_sum: Optional[torch.Tensor] = None
        last_stab: Optional[torch.Tensor] = None

        for indices in windows:
            burst = sequence[indices].unsqueeze(0)  # [1, N, 3, H, W]
            pred = self.predict_burst(model, burst, amp_dtype=amp_dtype)
            restored_sum = pred.restored if restored_sum is None else restored_sum + pred.restored
            anchor_sum = pred.anchor if anchor_sum is None else anchor_sum + pred.anchor
            residual_sum = pred.residual if residual_sum is None else residual_sum + pred.residual
            var_sum = pred.variance_map if var_sum is None else var_sum + pred.variance_map
            last_stab = pred.stabilized_frames

        count = float(len(windows))
        assert restored_sum is not None and anchor_sum is not None
        assert residual_sum is not None and var_sum is not None and last_stab is not None

        return ModelPrediction(
            restored=restored_sum / count,
            anchor=anchor_sum / count,
            residual=residual_sum / count,
            variance_map=var_sum / count,
            stabilized_frames=last_stab,
        )
