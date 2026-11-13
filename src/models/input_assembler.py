"""Input tensor assembly (Single Responsibility) supporting ablation modes A1..A3c."""

from typing import Tuple
import torch

from src.preprocessing.variance import compute_temporal_std


class InputAssembler:
    """
    Constructs the stacked input tensor X_in [B, C_in, H, W] from a stabilized burst
    according to `input_mode`, keeping channel counts synchronized with the stem conv.
    """

    VALID_MODES = ("mean_only", "frames_only", "frames_anchor", "full")

    def __init__(self, num_frames: int, input_mode: str = "full") -> None:
        if input_mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid input_mode '{input_mode}'. Expected one of {self.VALID_MODES}."
            )
        self.num_frames = int(num_frames)
        self.input_mode = input_mode

    @property
    def in_channels(self) -> int:
        if self.input_mode == "mean_only":
            return 3
        if self.input_mode == "frames_only":
            return 3 * self.num_frames
        if self.input_mode == "frames_anchor":
            return 3 * self.num_frames + 3
        return 3 * self.num_frames + 4

    def assemble(
        self, frames: torch.Tensor, anchor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            frames: [B, N, 3, H, W]
            anchor: [B, 3, H, W]
        Returns:
            x_in: [B, C_in, H, W]
            variance_map: [B, 1, H, W]
        """
        b, n, c, h, w = frames.shape
        if self.input_mode != "mean_only" and n != self.num_frames:
            raise ValueError(
                f"Expected {self.num_frames} frames for mode '{self.input_mode}', got {n}."
            )

        variance_map = compute_temporal_std(frames)

        if self.input_mode == "mean_only":
            return anchor, variance_map

        frames_flat = frames.reshape(b, n * c, h, w)
        if self.input_mode == "frames_only":
            return frames_flat, variance_map
        if self.input_mode == "frames_anchor":
            return torch.cat([frames_flat, anchor], dim=1), variance_map

        # full: [I_1 ... I_N, I_anchor, U_var]
        return torch.cat([frames_flat, anchor, variance_map], dim=1), variance_map
