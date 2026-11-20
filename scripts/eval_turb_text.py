"""Restore Turbulence-Text (UG2+ Track 3) scenes and export word crops for OCR evaluation."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

RESUNET_ROOT = Path(__file__).resolve().parents[1]
if str(RESUNET_ROOT) not in sys.path:
    sys.path.insert(0, str(RESUNET_ROOT))

from src.config import load_config, validate_config  # noqa: E402
from src.engine import AtomicCheckpointer, TiledInference  # noqa: E402
from src.engine.trainer import resolve_amp_config  # noqa: E402
from src.models import build_model  # noqa: E402
from src.preprocessing import build_anchor_builder  # noqa: E402
from src.registry import STABILIZER_REGISTRY  # noqa: E402

CLASSICAL_MODES = {"A0-mean": "mean", "A0-frf": "frf", "A0-lucky": "lucky"}
ALL_MODES = ["raw_center", "raw_mean", "A0-mean", "A0-frf", "A0-lucky", "neural"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore Turbulence-Text scenes and export cropped word images."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path containing final_test/ and labels_final.csv.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help="Optional override path to labels_final.csv.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=ALL_MODES,
        help="Restoration mode to run.",
    )
    parser.add_argument(
        "--method-name",
        type=str,
        default=None,
        help="Output subfolder name (defaults to --mode).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path (required when --mode neural).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=RESUNET_ROOT / "configs" / "default.yaml",
        help="Fallback YAML config path.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=24,
        help="Window size for raw_mean and A0-* modes (neural mode uses checkpoint config).",
    )
    parser.add_argument(
        "--num-windows",
        type=int,
        default=4,
        help="Number of temporal windows averaged per scene.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=0,
        help="Optional intermediate square resize before restoration (e.g. 360 like DATUM). 0 keeps native 440x440.",
    )
    parser.add_argument(
        "--stabilize",
        action="store_true",
        help="Enable FFTGlobalStabilizer before restoration.",
    )
    parser.add_argument(
        "--per-window",
        action="store_true",
        help="Also export crops for each temporal window separately (4x500=2000 crops).",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=0,
        help="Optional limit on number of scenes (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESUNET_ROOT / "eval_vis" / "turb_text",
        help="Root output directory for restored scenes, crops, and manifest.json.",
    )
    return parser.parse_args()


def load_labels(csv_path: Path) -> Dict[int, List[Dict[str, Any]]]:
    """Parse labels_final.csv into {scene_idx: [{word_idx, text, box}, ...]}."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Labels CSV not found: {csv_path}")

    by_scene: Dict[int, List[Dict[str, Any]]] = {}
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        _header = next(reader, None)
        for row in reader:
            if not row or len(row) < 6:
                continue
            scene_id = int(row[0].strip())
            text = row[1].strip()
            box = (int(row[2]), int(row[3]), int(row[4]), int(row[5]))
            items = by_scene.setdefault(scene_id, [])
            items.append({"word_idx": len(items) + 1, "text": text, "box": box})
    return by_scene


def load_scene_sequence(scene_dir: Path) -> torch.Tensor:
    """Load sorted PNG frames `[T, 3, H, W]` in `[0, 1]`, ignoring hidden files like .DS_Store."""
    frame_paths = sorted(p for p in scene_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png")
    if not frame_paths:
        raise RuntimeError(f"No PNG frames found in {scene_dir}")
    arrays = [np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8) for p in frame_paths]
    stacked = np.stack(arrays, axis=0)  # [T, H, W, 3]
    return torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous().float() / 255.0


def resize_sequence(seq: torch.Tensor, size: int) -> torch.Tensor:
    if size <= 0 or (seq.shape[-2] == size and seq.shape[-1] == size):
        return seq
    return F.interpolate(seq, size=(size, size), mode="bilinear", align_corners=False)


def resize_image(img: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if img.shape[-2] == height and img.shape[-1] == width:
        return img
    return F.interpolate(img, size=(height, width), mode="bilinear", align_corners=False)


def tensor_to_pil(img: torch.Tensor) -> Image.Image:
    arr = img.detach().cpu().float().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0
    return Image.fromarray(arr.round().astype(np.uint8))


def _restore_scene_windows(
    seq: torch.Tensor,
    mode: str,
    num_frames: int,
    evaluator: TiledInference,
    device: torch.device,
    model: Optional[torch.nn.Module],
    amp_dtype: Optional[torch.dtype],
    anchor_builder: Optional[torch.nn.Module],
    stabilizer: torch.nn.Module,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Return `(mean_restored [1, 3, H, W], per_window_restored_list)` at `seq` resolution."""
    total_frames = int(seq.shape[0])
    if mode == "raw_center":
        center = seq[total_frames // 2 : total_frames // 2 + 1]
        return center, [center]

    windows = evaluator.select_eval_windows(total_frames, num_frames)
    per_window: List[torch.Tensor] = []

    if mode == "neural":
        assert model is not None
        for win_indices in windows:
            burst = seq[win_indices].unsqueeze(0)
            pred = evaluator.predict_burst(model, burst, amp_dtype=amp_dtype)
            per_window.append(pred.restored.float())
    elif mode == "raw_mean":
        for win_indices in windows:
            burst = seq[win_indices].unsqueeze(0)
            if evaluator.stabilize_in_eval:
                burst = stabilizer.stabilize(burst)
            per_window.append(burst.mean(dim=1).float())
    else:
        assert anchor_builder is not None
        for win_indices in windows:
            burst = seq[win_indices].unsqueeze(0)
            if evaluator.stabilize_in_eval:
                burst = stabilizer.stabilize(burst)
            per_window.append(anchor_builder.build(burst).float())

    mean_img = torch.stack(per_window, dim=0).mean(dim=0)
    return mean_img, per_window


def _export_scene_and_crops(
    pil_scene: Image.Image,
    scene_name: str,
    annotations: List[Dict[str, Any]],
    scenes_dir: Path,
    crops_dir: Path,
    suffix: str = "",
) -> List[Dict[str, Any]]:
    scene_filename = f"{scene_name}{suffix}.png"
    pil_scene.save(scenes_dir / scene_filename)

    entries: List[Dict[str, Any]] = []
    for ann in annotations:
        x1, y1, x2, y2 = ann["box"]
        crop_pil = pil_scene.crop((x1, y1, x2, y2))
        crop_filename = f"{scene_name}{suffix}_w{ann['word_idx']:02d}.png"
        crop_path = crops_dir / crop_filename
        crop_pil.save(crop_path)
        entries.append(
            {
                "scene": scene_name,
                "window": suffix.lstrip("_") or "mean",
                "word_idx": ann["word_idx"],
                "text": ann["text"],
                "box": [x1, y1, x2, y2],
                "crop_path": str(crop_path),
            }
        )
    return entries


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    frames_root = data_root / "final_test"
    if not frames_root.is_dir():
        raise FileNotFoundError(f"Expected final_test/ directory under {data_root}")
    labels_csv = args.labels_csv or (data_root / "labels_final.csv")
    labels_by_scene = load_labels(labels_csv)

    method_name = args.method_name or args.mode
    out_method_dir = args.output_dir / method_name
    scenes_dir = out_method_dir / "scenes"
    crops_dir = out_method_dir / "crops"
    per_win_scenes_dir = out_method_dir / "scenes_per_window"
    per_win_crops_dir = out_method_dir / "crops_per_window"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    if args.per_window:
        per_win_scenes_dir.mkdir(parents=True, exist_ok=True)
        per_win_crops_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_cfg = load_config(args.config, variant=None)

    model: Optional[torch.nn.Module] = None
    amp_dtype: Optional[torch.dtype] = None
    anchor_builder: Optional[torch.nn.Module] = None

    if args.mode == "neural":
        if args.checkpoint is None or not args.checkpoint.is_file():
            raise FileNotFoundError(f"Valid --checkpoint is required for neural mode: {args.checkpoint}")
        try:
            peek = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            peek = torch.load(args.checkpoint, map_location="cpu")
        cfg = copy.deepcopy(peek["config"]) if isinstance(peek, dict) and isinstance(peek.get("config"), dict) else copy.deepcopy(base_cfg)
        cfg = validate_config(cfg)
        model = build_model(cfg).to(device)
        AtomicCheckpointer(args.checkpoint.parent).load(args.checkpoint, model=model, map_location=str(device))
        model.eval()
        num_frames = int(cfg["model"]["num_frames"])
        _, amp_dtype, _ = resolve_amp_config(device, str(cfg["optim"].get("amp_dtype", "float32")).lower())
    else:
        cfg = validate_config(copy.deepcopy(base_cfg))
        num_frames = int(args.num_frames)
        if args.mode in CLASSICAL_MODES:
            anchor_builder = build_anchor_builder(
                cfg["model"], anchor_override=CLASSICAL_MODES[args.mode]
            ).to(device)

    stabilizer = (
        STABILIZER_REGISTRY.build("fft_global").to(device)
        if args.stabilize
        else STABILIZER_REGISTRY.build("none").to(device)
    )
    evaluator = TiledInference(
        tile_size=int(cfg["eval"].get("tile_size", 256)),
        overlap=int(cfg["eval"].get("tile_overlap", 64)),
        max_eval_windows=int(args.num_windows),
        stabilize_in_eval=bool(args.stabilize),
    )

    scene_dirs = sorted(d for d in frames_root.iterdir() if d.is_dir())
    if args.max_scenes > 0:
        scene_dirs = scene_dirs[: args.max_scenes]

    manifest_mean: List[Dict[str, Any]] = []
    manifest_per_window: List[Dict[str, Any]] = []

    for idx, scene_dir in enumerate(scene_dirs, 1):
        scene_num = int(scene_dir.name.split("_")[-1])
        annotations = labels_by_scene.get(scene_num, [])
        if not annotations:
            raise RuntimeError(f"No annotations found for scene {scene_dir.name} (#{scene_num})")

        seq = load_scene_sequence(scene_dir).to(device)
        orig_h, orig_w = int(seq.shape[-2]), int(seq.shape[-1])
        work_seq = resize_sequence(seq, args.resize) if args.resize > 0 else seq

        mean_img, per_win_imgs = _restore_scene_windows(
            seq=work_seq,
            mode=args.mode,
            num_frames=num_frames,
            evaluator=evaluator,
            device=device,
            model=model,
            amp_dtype=amp_dtype,
            anchor_builder=anchor_builder,
            stabilizer=stabilizer,
        )

        mean_img = resize_image(mean_img, orig_h, orig_w)
        manifest_mean.extend(
            _export_scene_and_crops(
                pil_scene=tensor_to_pil(mean_img[0]),
                scene_name=scene_dir.name,
                annotations=annotations,
                scenes_dir=scenes_dir,
                crops_dir=crops_dir,
            )
        )

        if args.per_window:
            for win_idx, win_img in enumerate(per_win_imgs, 1):
                win_img = resize_image(win_img, orig_h, orig_w)
                manifest_per_window.extend(
                    _export_scene_and_crops(
                        pil_scene=tensor_to_pil(win_img[0]),
                        scene_name=scene_dir.name,
                        annotations=annotations,
                        scenes_dir=per_win_scenes_dir,
                        crops_dir=per_win_crops_dir,
                        suffix=f"_win{win_idx:02d}",
                    )
                )

        if idx % 10 == 0 or idx == len(scene_dirs):
            print(f"[{method_name}] Processadas {idx}/{len(scene_dirs)} cenas ({len(manifest_mean)} crops)")

    manifest_payload = {
        "method_name": method_name,
        "mode": args.mode,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "num_frames": num_frames,
        "num_windows": args.num_windows,
        "resize": args.resize,
        "stabilize": bool(args.stabilize),
        "num_scenes": len(scene_dirs),
        "num_crops": len(manifest_mean),
        "crops": manifest_mean,
    }
    manifest_path = out_method_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)

    if args.per_window:
        per_win_payload = dict(manifest_payload)
        per_win_payload["num_crops"] = len(manifest_per_window)
        per_win_payload["crops"] = manifest_per_window
        with open(out_method_dir / "manifest_per_window.json", "w", encoding="utf-8") as handle:
            json.dump(per_win_payload, handle, indent=2)

    print(f"[{method_name}] Concluido. Manifesto salvo em {manifest_path}")


if __name__ == "__main__":
    main()
