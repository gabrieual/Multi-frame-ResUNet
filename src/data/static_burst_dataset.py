"""ATSyn-static dataset loader with mmap .npy cache support and strict safety checks."""

from dataclasses import dataclass
from pathlib import Path
import random
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.contracts import BurstBatch
from src.data.transforms import SyncedCropAndAugment

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    turb_path: Path   # Directory of frame images OR `turb.npy`
    gt_path: Path     # Directory containing 1 image, direct image file, OR `gt.npy`
    is_npy: bool


def _list_images(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    )


def _find_matching_gt(gt_root: Path, stem: str) -> Optional[Path]:
    direct = gt_root / stem
    if direct.is_file() and direct.suffix.lower() in IMG_EXTENSIONS:
        return direct
    if direct.is_dir() and len(_list_images(direct)) > 0:
        return direct
    for ext in IMG_EXTENSIONS:
        for cand in (gt_root / f"{stem}{ext}", gt_root / f"{stem}{ext.upper()}"):
            if cand.is_file():
                return cand
    return None


class StaticBurstDataset(Dataset):
    """
    Loads static turbulence bursts -> single ground truth image from ATSyn-static.

    Supports three on-disk layouts transparently:
      1. Fast pre-converted mmap cache: `<root>/<sample_id>/{turb.npy, gt.npy}`
      2. Per-sample directory: `<root>/<sample_id>/{turb/*.png, gt/*.png}`
      3. Modality-split layout: `<root>/{turb/<sample_id>/*.png, gt/<sample_id>[.png|/]}`

    Temporal sampling in training randomly picks N frames WITHOUT sorting their indices,
    preserving random channel ordering to encourage permutation robustness.
    """

    def __init__(
        self,
        root_dir: str | Path,
        num_frames: int = 8,
        crop_size: int = 256,
        is_train: bool = True,
        allow_synthetic_fallback: bool = False,
        return_all_frames: bool = False,
    ) -> None:
        super().__init__()
        if allow_synthetic_fallback:
            raise ValueError(
                "Synthetic fallback is strictly disabled in StaticBurstDataset."
            )

        raw_root = Path(root_dir)
        if not raw_root.exists() and raw_root.name == "val" and (raw_root.parent / "test").exists():
            raw_root = raw_root.parent / "test"

        if not raw_root.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {raw_root}")

        self.root_dir = raw_root
        self.num_frames = int(num_frames)
        self.is_train = bool(is_train)
        self.return_all_frames = bool(return_all_frames)
        self.transform = SyncedCropAndAugment(crop_size=crop_size, is_train=is_train)

        self.samples: List[SampleRecord] = self._discover_samples(self.root_dir)
        if not self.samples:
            raise FileNotFoundError(
                f"No valid static burst samples found under: {self.root_dir}"
            )

    @staticmethod
    def _discover_samples(root: Path) -> List[SampleRecord]:
        records: List[SampleRecord] = []

        # Layout 3: Modality-split `<root>/{turb, degraded}/<sample>` and `<root>/gt/<sample>`
        turb_root = (root / "turb") if (root / "turb").exists() else (root / "degraded")
        gt_root = root / "gt"
        if turb_root.exists() and gt_root.exists():
            for entry in sorted(turb_root.iterdir()):
                if not entry.is_dir():
                    continue
                gt_match = _find_matching_gt(gt_root, entry.name)
                if gt_match is not None and len(_list_images(entry)) > 0:
                    records.append(
                        SampleRecord(
                            sample_id=entry.name,
                            turb_path=entry,
                            gt_path=gt_match,
                            is_npy=False,
                        )
                    )
                elif (gt_root / entry.name).is_dir():
                    # Nested category folder `<root>/turb/<cat>/<sample>`
                    cat_gt_root = gt_root / entry.name
                    for sub_entry in sorted(entry.iterdir()):
                        if sub_entry.is_dir() and len(_list_images(sub_entry)) > 0:
                            sub_gt = _find_matching_gt(cat_gt_root, sub_entry.name)
                            if sub_gt is not None:
                                records.append(
                                    SampleRecord(
                                        sample_id=f"{entry.name}_{sub_entry.name}",
                                        turb_path=sub_entry,
                                        gt_path=sub_gt,
                                        is_npy=False,
                                    )
                                )
            if records:
                return records

        # Layout 1 (.npy cache) & Layout 2 (per-sample subdirectories)
        for sample_dir in sorted(root.iterdir()):
            if not sample_dir.is_dir():
                continue
            turb_npy = sample_dir / "turb.npy"
            gt_npy = sample_dir / "gt.npy"
            if turb_npy.is_file() and gt_npy.is_file():
                records.append(
                    SampleRecord(
                        sample_id=sample_dir.name,
                        turb_path=turb_npy,
                        gt_path=gt_npy,
                        is_npy=True,
                    )
                )
                continue

            sub_turb = (sample_dir / "turb") if (sample_dir / "turb").exists() else (sample_dir / "degraded")
            sub_gt = _find_matching_gt(sample_dir, "gt")
            if sub_turb.is_dir() and sub_gt is not None and len(_list_images(sub_turb)) > 0:
                gt_file = sub_gt if sub_gt.is_file() else _list_images(sub_gt)[0]
                records.append(
                    SampleRecord(
                        sample_id=sample_dir.name,
                        turb_path=sub_turb,
                        gt_path=gt_file,
                        is_npy=False,
                    )
                )

        return records

    def _select_indices(self, total_frames: int) -> List[int]:
        if total_frames <= 0:
            raise ValueError("Sample contains 0 frames.")
        if self.return_all_frames:
            return list(range(total_frames))

        if self.is_train:
            if total_frames >= self.num_frames:
                # Random permutation order without sorting (ResUNetMultiFrame.md §4)
                return random.sample(range(total_frames), self.num_frames)
            return random.choices(range(total_frames), k=self.num_frames)

        # Deterministic uniform sampling for repeatable validation/evaluation
        return np.linspace(0, total_frames - 1, self.num_frames, dtype=int).tolist()

    @staticmethod
    def _load_image_tensor(path: Path) -> torch.Tensor:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            arr = np.asarray(rgb, dtype=np.uint8)
        return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float().div_(255.0)

    def _load_sample_tensors(self, rec: SampleRecord) -> Tuple[torch.Tensor, torch.Tensor]:
        if rec.is_npy:
            turb_mmap = np.load(rec.turb_path, mmap_mode="r")
            indices = self._select_indices(int(turb_mmap.shape[0]))
            # Read only selected frames from mmap into contiguous float tensor [N, 3, H, W]
            frames_np = np.ascontiguousarray(turb_mmap[indices])
            gt_np = np.ascontiguousarray(np.load(rec.gt_path))
            if frames_np.ndim == 4 and frames_np.shape[-1] == 3:
                frames = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float()
            else:
                frames = torch.from_numpy(frames_np).float()
            if frames_np.dtype == np.uint8 or frames.max() > 1.0:
                frames = frames.div_(255.0)

            if gt_np.ndim == 4:
                gt_np = np.ascontiguousarray(gt_np[0])
            if gt_np.ndim == 3 and gt_np.shape[-1] == 3:
                gt = torch.from_numpy(gt_np).permute(2, 0, 1).float()
            else:
                gt = torch.from_numpy(gt_np).float()
            if gt_np.dtype == np.uint8 or gt.max() > 1.0:
                gt = gt.div_(255.0)
            return frames, gt

        frame_paths = _list_images(rec.turb_path)
        indices = self._select_indices(len(frame_paths))
        frames = torch.stack([self._load_image_tensor(frame_paths[i]) for i in indices], dim=0)

        if rec.gt_path.is_dir():
            gt_candidates = _list_images(rec.gt_path)
            gt_file = gt_candidates[0]
        else:
            gt_file = rec.gt_path
        gt = self._load_image_tensor(gt_file)
        return frames, gt

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        rec = self.samples[idx]
        frames, gt = self._load_sample_tensors(rec)
        frames, gt = self.transform(frames, gt)
        return frames, gt, rec.sample_id


def burst_collate_fn(
    items: List[Tuple[torch.Tensor, torch.Tensor, str]],
) -> BurstBatch:
    degraded = torch.stack([it[0] for it in items], dim=0)
    gt = torch.stack([it[1] for it in items], dim=0)
    sample_ids = tuple(it[2] for it in items)
    return BurstBatch(degraded=degraded, gt=gt, sample_ids=sample_ids)
