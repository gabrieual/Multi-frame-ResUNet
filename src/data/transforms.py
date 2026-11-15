"""Synchronized spatial cropping and geometric augmentations for static bursts."""

import random
from typing import Tuple
import torch
import torch.nn.functional as F


class SyncedCropAndAugment:
    """
    Applies identical spatial crop, flips, and 90-degree rotations to all N burst
    frames [N, 3, H, W] and the single clean ground-truth image [3, H, W].
    """

    def __init__(self, crop_size: int = 256, is_train: bool = True) -> None:
        self.crop_size = int(crop_size)
        self.is_train = bool(is_train)

    @staticmethod
    def _pad_if_needed(
        frames: torch.Tensor, gt: torch.Tensor, min_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, h, w = frames.shape
        pad_h = max(0, min_size - h)
        pad_w = max(0, min_size - w)
        if pad_h == 0 and pad_w == 0:
            return frames, gt
        mode = "reflect" if (h > pad_h and w > pad_w) else "replicate"
        frames = F.pad(frames, (0, pad_w, 0, pad_h), mode=mode)
        gt = F.pad(gt.unsqueeze(0), (0, pad_w, 0, pad_h), mode=mode).squeeze(0)
        return frames, gt

    def augment_only(
        self, frames: torch.Tensor, gt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies random flips and 90-degree rotations without cropping."""
        if self.is_train:
            if random.random() < 0.5:
                frames = torch.flip(frames, dims=[-1])
                gt = torch.flip(gt, dims=[-1])
            if random.random() < 0.5:
                frames = torch.flip(frames, dims=[-2])
                gt = torch.flip(gt, dims=[-2])
            k = random.randint(0, 3)
            if k > 0:
                frames = torch.rot90(frames, k=k, dims=[-2, -1])
                gt = torch.rot90(gt, k=k, dims=[-2, -1])
        return frames, gt

    def __call__(
        self, frames: torch.Tensor, gt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            frames: [N, 3, H, W]
            gt: [3, H, W]
        Returns:
            (cropped_frames, cropped_gt)
        """
        if self.crop_size > 0:
            frames, gt = self._pad_if_needed(frames, gt, self.crop_size)
            _, _, h, w = frames.shape
            if self.is_train:
                top = random.randint(0, h - self.crop_size)
                left = random.randint(0, w - self.crop_size)
            else:
                top = (h - self.crop_size) // 2
                left = (w - self.crop_size) // 2

            frames = frames[:, :, top : top + self.crop_size, left : left + self.crop_size]
            gt = gt[:, top : top + self.crop_size, left : left + self.crop_size]

        frames, gt = self.augment_only(frames, gt)
        return frames.contiguous(), gt.contiguous()
