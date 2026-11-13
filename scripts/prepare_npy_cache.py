"""Convert ATSyn-static PNG bursts into uint8 `.npy` mmap cache to eliminate I/O bottlenecks."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
from typing import Optional, Tuple

import numpy as np
from PIL import Image

RESUNET_ROOT = Path(__file__).resolve().parents[1]
if str(RESUNET_ROOT) not in sys.path:
    sys.path.insert(0, str(RESUNET_ROOT))

from src.data.static_burst_dataset import SampleRecord, StaticBurstDataset, _list_images  # noqa: E402


def _read_rgb_uint8(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _convert_sample(
    record: SampleRecord,
    out_root: Path,
    overwrite: bool = False,
) -> Tuple[str, int, bool]:
    """Convert one sample directory into `turb.npy` [T, H, W, 3] and `gt.npy` [H, W, 3]."""
    sample_out = out_root / record.sample_id
    turb_dst = sample_out / "turb.npy"
    gt_dst = sample_out / "gt.npy"

    if not overwrite and turb_dst.is_file() and gt_dst.is_file():
        arr = np.load(turb_dst, mmap_mode="r")
        return record.sample_id, int(arr.shape[0]), False

    if record.is_npy:
        turb_arr = np.asarray(np.load(record.turb_path))
        gt_arr = np.asarray(np.load(record.gt_path))
    else:
        frame_paths = _list_images(record.turb_path)
        frames = [_read_rgb_uint8(p) for p in frame_paths]
        turb_arr = np.stack(frames, axis=0)  # [T, H, W, 3] uint8

        gt_file = _list_images(record.gt_path)[0] if record.gt_path.is_dir() else record.gt_path
        gt_arr = _read_rgb_uint8(gt_file)  # [H, W, 3] uint8

    sample_out.mkdir(parents=True, exist_ok=True)
    np.save(turb_dst, turb_arr)
    np.save(gt_dst, gt_arr)
    return record.sample_id, int(turb_arr.shape[0]), True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-convert static PNG bursts to uint8 .npy mmap caches."
    )
    parser.add_argument("--src", type=Path, required=True, help="Source dataset split directory")
    parser.add_argument("--dst", type=Path, required=True, help="Destination .npy cache directory")
    parser.add_argument("--workers", type=int, default=4, help="Parallel I/O conversion threads")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap (0 = all)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .npy files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = StaticBurstDataset._discover_samples(args.src)
    if not records:
        raise FileNotFoundError(f"No valid static burst samples found under: {args.src}")

    if args.max_samples > 0:
        records = records[: args.max_samples]

    args.dst.mkdir(parents=True, exist_ok=True)
    converted = 0
    skipped = 0
    total_frames = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(_convert_sample, rec, args.dst, args.overwrite) for rec in records
        ]
        for fut in as_completed(futures):
            _, num_frames, did_write = fut.result()
            total_frames += num_frames
            if did_write:
                converted += 1
            else:
                skipped += 1

    print(
        f"Done: {converted} converted, {skipped} skipped, "
        f"{total_frames} total frames written to {args.dst}"
    )


if __name__ == "__main__":
    main()
