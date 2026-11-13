"""Official full-image Overlap-Tile (256x256) evaluator for baselines A0 and neural A1..A3c."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
import torch

RESUNET_ROOT = Path(__file__).resolve().parents[1]
if str(RESUNET_ROOT) not in sys.path:
    sys.path.insert(0, str(RESUNET_ROOT))

from src.config import ABLATION_PRESETS, apply_variant_preset, load_config, validate_config  # noqa: E402
from src.data import StaticBurstDataset  # noqa: E402
from src.engine import AtomicCheckpointer, TiledInference  # noqa: E402
from src.engine.trainer import resolve_amp_config  # noqa: E402
from src.metrics import compute_psnr, compute_ssim  # noqa: E402
from src.models import build_model  # noqa: E402
import src.preprocessing  # noqa: E402, F401
from src.registry import ANCHOR_REGISTRY, STABILIZER_REGISTRY  # noqa: E402

EVAL_VARIANTS = sorted(list(ABLATION_PRESETS.keys()) + ["A0-mean", "A0-frf"])


def _tensor_to_pil(img: torch.Tensor) -> Image.Image:
    arr = (img.detach().cpu().float().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0)
    return Image.fromarray(arr.round().astype(np.uint8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-image Overlap-Tile (256x256) evaluator for static turbulence bursts."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=RESUNET_ROOT / "configs" / "default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        choices=EVAL_VARIANTS,
        help="Variant to evaluate (`A0-mean`, `A0-frf`, or `A1..A3c`).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Model checkpoint path (required for neural variants A1..A3c).",
    )
    parser.add_argument("--test-root", type=str, default=None, help="Override data.val_root")
    parser.add_argument("--tile-size", type=int, default=None, help="Override eval.tile_size (256)")
    parser.add_argument("--tile-overlap", type=int, default=None, help="Override eval.tile_overlap")
    parser.add_argument(
        "--max-eval-windows", type=int, default=None, help="Override eval.max_eval_windows"
    )
    parser.add_argument("--stabilize", action="store_true", help="Enable global FFT shift registration")
    parser.add_argument("--save-dir", type=Path, default=None, help="Optional dir for PNG outputs")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON report path")
    return parser.parse_args()


def _evaluate_classical_anchor(
    variant: str,
    dataset: StaticBurstDataset,
    evaluator: TiledInference,
    cfg: Dict[str, Any],
    device: torch.device,
    save_dir: Optional[Path],
) -> List[Dict[str, Any]]:
    anchor_name = "mean" if variant == "A0-mean" else "frf"
    model_cfg = cfg["model"]
    if anchor_name == "frf":
        anchor_builder = ANCHOR_REGISTRY.build(
            anchor_name,
            bandwidth_h=float(model_cfg.get("frf_bandwidth_h", 0.05)),
            sigma=float(model_cfg.get("frf_sigma", 0.5)),
        )
    else:
        anchor_builder = ANCHOR_REGISTRY.build(anchor_name)
    stabilizer = (
        STABILIZER_REGISTRY.build("fft_global")
        if evaluator.stabilize_in_eval
        else STABILIZER_REGISTRY.build("none")
    )
    num_frames = int(cfg["data"]["num_frames"])

    results: List[Dict[str, Any]] = []
    for idx in range(len(dataset)):
        seq, gt, sample_id = dataset[idx]
        seq = seq.to(device)
        gt_b = gt.unsqueeze(0).to(device)
        windows = evaluator.select_eval_windows(int(seq.shape[0]), num_frames)

        acc: Optional[torch.Tensor] = None
        for win_indices in windows:
            burst = seq[win_indices].unsqueeze(0)
            if evaluator.stabilize_in_eval:
                burst = stabilizer.stabilize(burst)
            anchor = anchor_builder.build(burst).float()
            acc = anchor if acc is None else acc + anchor

        assert acc is not None
        pred = acc / float(len(windows))
        psnr = compute_psnr(pred, gt_b)
        ssim = compute_ssim(pred, gt_b)
        results.append(
            {
                "sample_id": sample_id,
                "num_frames": int(seq.shape[0]),
                "psnr": psnr,
                "anchor_psnr": psnr,
                "psnr_gain": 0.0,
                "ssim": ssim,
            }
        )
        if save_dir is not None:
            sample_out = save_dir / sample_id
            sample_out.mkdir(parents=True, exist_ok=True)
            _tensor_to_pil(pred[0]).save(sample_out / f"{variant}.png")
    return results


def _evaluate_neural_model(
    cfg: Dict[str, Any],
    checkpoint_path: Path,
    dataset: StaticBurstDataset,
    evaluator: TiledInference,
    device: torch.device,
    save_dir: Optional[Path],
) -> List[Dict[str, Any]]:
    checkpointer = AtomicCheckpointer(checkpoint_path.parent)
    model = build_model(cfg).to(device)
    checkpointer.load(checkpoint_path, model=model, map_location=str(device))
    model.eval()

    num_frames = int(cfg["model"]["num_frames"])
    amp_str = str(cfg["optim"].get("amp_dtype", "float32")).lower()
    _, amp_dtype, _ = resolve_amp_config(device, amp_str)

    results: List[Dict[str, Any]] = []
    for idx in range(len(dataset)):
        seq, gt, sample_id = dataset[idx]
        seq = seq.to(device)
        gt_b = gt.unsqueeze(0).to(device)
        pred = evaluator.predict_sequence(model, seq, num_frames=num_frames, amp_dtype=amp_dtype)

        restored_psnr = compute_psnr(pred.restored, gt_b)
        anchor_psnr = compute_psnr(pred.anchor, gt_b)
        restored_ssim = compute_ssim(pred.restored, gt_b)
        results.append(
            {
                "sample_id": sample_id,
                "num_frames": int(seq.shape[0]),
                "psnr": restored_psnr,
                "anchor_psnr": anchor_psnr,
                "psnr_gain": restored_psnr - anchor_psnr,
                "ssim": restored_ssim,
            }
        )
        if save_dir is not None:
            sample_out = save_dir / sample_id
            sample_out.mkdir(parents=True, exist_ok=True)
            _tensor_to_pil(pred.restored[0]).save(sample_out / "restored.png")
            strip = torch.cat([pred.anchor[0], pred.restored[0], gt_b[0]], dim=2)
            _tensor_to_pil(strip).save(sample_out / "comparison_anchor_restored_gt.png")
    return results


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.config, variant=None)

    variant = args.variant
    is_classical = variant in {"A0-mean", "A0-frf"}

    if not is_classical:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for neural variants A1..A3c.")
        if not args.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        try:
            peek = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            peek = torch.load(args.checkpoint, map_location="cpu")
        if isinstance(peek, dict) and isinstance(peek.get("config"), dict):
            cfg = copy.deepcopy(peek["config"])
        else:
            cfg = copy.deepcopy(base_cfg)
        if variant is not None:
            cfg = apply_variant_preset(cfg, variant)
    else:
        cfg = copy.deepcopy(base_cfg)

    if args.test_root is not None:
        cfg["data"]["val_root"] = args.test_root
    if args.tile_size is not None:
        cfg["eval"]["tile_size"] = args.tile_size
    if args.tile_overlap is not None:
        cfg["eval"]["tile_overlap"] = args.tile_overlap
    if args.max_eval_windows is not None:
        cfg["eval"]["max_eval_windows"] = args.max_eval_windows
    if args.stabilize:
        cfg["model"]["stabilize_in_eval"] = True

    cfg = validate_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_root = cfg["data"]["val_root"]
    dataset = StaticBurstDataset(
        root_dir=test_root,
        num_frames=int(cfg["data"]["num_frames"]),
        crop_size=0,  # Full uncropped image resolution for official Overlap-Tile eval
        is_train=False,
        allow_synthetic_fallback=bool(cfg["data"].get("allow_synthetic_fallback", False)),
        return_all_frames=True,
    )

    eval_cfg = cfg["eval"]
    evaluator = TiledInference(
        tile_size=int(eval_cfg.get("tile_size", 256)),
        overlap=int(eval_cfg.get("tile_overlap", 64)),
        max_eval_windows=int(eval_cfg.get("max_eval_windows", 4)),
        stabilize_in_eval=bool(cfg["model"].get("stabilize_in_eval", False)),
    )

    if is_classical:
        assert variant is not None
        sample_results = _evaluate_classical_anchor(
            variant=variant,
            dataset=dataset,
            evaluator=evaluator,
            cfg=cfg,
            device=device,
            save_dir=args.save_dir,
        )
        label = variant
    else:
        assert args.checkpoint is not None
        sample_results = _evaluate_neural_model(
            cfg=cfg,
            checkpoint_path=args.checkpoint,
            dataset=dataset,
            evaluator=evaluator,
            device=device,
            save_dir=args.save_dir,
        )
        label = str(cfg["experiment"].get("variant", "custom"))

    mean_psnr = float(np.mean([r["psnr"] for r in sample_results]))
    mean_anchor = float(np.mean([r["anchor_psnr"] for r in sample_results]))
    mean_gain = float(np.mean([r["psnr_gain"] for r in sample_results]))
    mean_ssim = float(np.mean([r["ssim"] for r in sample_results]))

    print(f"\n=== Evaluation Summary ({label} | {len(sample_results)} samples | Tile {evaluator.tile_size}x{evaluator.tile_size}) ===")
    for row in sample_results:
        print(
            f"  {row['sample_id']:24s} | T={row['num_frames']:3d} | "
            f"Anchor={row['anchor_psnr']:6.2f}dB | Restored={row['psnr']:6.2f}dB "
            f"({row['psnr_gain']:+5.2f}dB) | SSIM={row['ssim']:.4f}"
        )
    print(
        f"  {'MEAN':24s} |       | "
        f"Anchor={mean_anchor:6.2f}dB | Restored={mean_psnr:6.2f}dB "
        f"({mean_gain:+5.2f}dB) | SSIM={mean_ssim:.4f}\n"
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "variant": label,
            "tile_size": evaluator.tile_size,
            "tile_overlap": evaluator.overlap,
            "max_eval_windows": evaluator.max_eval_windows,
            "stabilize_in_eval": evaluator.stabilize_in_eval,
            "num_samples": len(sample_results),
            "mean_psnr": mean_psnr,
            "mean_anchor_psnr": mean_anchor,
            "mean_psnr_gain": mean_gain,
            "mean_ssim": mean_ssim,
            "samples": sample_results,
        }
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
