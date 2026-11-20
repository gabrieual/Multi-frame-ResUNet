"""Unified evaluator for real-world turbulence datasets (OTIS, BVI-CLEAR, RLR-AT).

Saves per-scene metrics (PSNR/SSIM/CW-SSIM when GT is available, plus Tenengrad and
Laplacian sharpness for all scenes) to CSV/JSON and generates labeled side-by-side visual
comparison strips ready for inclusion in reports.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

RESUNET_ROOT = Path(__file__).resolve().parents[1]
if str(RESUNET_ROOT) not in sys.path:
    sys.path.insert(0, str(RESUNET_ROOT))

from src.config import load_config, validate_config  # noqa: E402
from src.engine import AtomicCheckpointer, TiledInference  # noqa: E402
from src.engine.trainer import resolve_amp_config  # noqa: E402
from src.metrics import compute_cw_ssim, compute_psnr, compute_ssim  # noqa: E402
from src.models import build_model  # noqa: E402
from src.preprocessing import build_anchor_builder  # noqa: E402
from src.registry import STABILIZER_REGISTRY  # noqa: E402


@dataclass
class RealSceneSpec:
    dataset: str
    scene_id: str
    category: str
    source_type: str  # "png_dir" or "mp4"
    source_path: Path
    gt_path: Optional[Path] = None
    mask_path: Optional[Path] = None
    clear_ref_path: Optional[Path] = None


@dataclass
class LoadedModel:
    name: str
    model: torch.nn.Module
    num_frames: int
    amp_dtype: Optional[torch.dtype]
    stabilize: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baselines and ResUNet checkpoints on OTIS, BVI-CLEAR, and RLR-AT."
    )
    parser.add_argument("--otis-root", type=Path, default=None, help="Path to OTIS root (OTIS_PNG_Color).")
    parser.add_argument("--bvi-root", type=Path, default=None, help="Path to BVI-CLEAR root.")
    parser.add_argument("--rlr-root", type=Path, default=None, help="Path to RLR-AT root.")
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=[],
        help="Named checkpoints as `label=/path/to/ckpt.pt`.",
    )
    parser.add_argument(
        "--also-stabilize",
        action="store_true",
        help="Also run a `<label>_stab` pass with FFTGlobalStabilizer for each neural checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=RESUNET_ROOT / "configs" / "default.yaml",
        help="Default YAML config for classical baselines.",
    )
    parser.add_argument(
        "--classical-frames",
        type=int,
        default=24,
        help="Frame window length for raw_mean and A0-lucky baselines.",
    )
    parser.add_argument(
        "--num-windows",
        type=int,
        default=4,
        help="Temporal windows averaged on static PNG datasets (OTIS, BVI-CLEAR).",
    )
    parser.add_argument(
        "--rlr-fps-stride",
        type=int,
        default=1,
        help="Frame stride for the consecutive center burst in RLR-AT videos.",
    )
    parser.add_argument(
        "--include-otis-moving",
        action="store_true",
        help="Include OTIS Moving Target/Car1..4 sequences.",
    )
    parser.add_argument(
        "--max-scenes-per-dataset",
        type=int,
        default=0,
        help="Optional limit per dataset (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--strip-max-width",
        type=int,
        default=640,
        help="Max per-panel width in overview strips (0 keeps native resolution).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESUNET_ROOT / "eval_vis" / "real_datasets",
        help="Output folder for restored images, strips, CSV, and JSON.",
    )
    return parser.parse_args()


def _sanitize_id(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name).strip("_")


def _discover_otis(root: Path, include_moving: bool) -> List[RealSceneSpec]:
    base = root / "OTIS_PNG_Color" if (root / "OTIS_PNG_Color").is_dir() else root
    specs: List[RealSceneSpec] = []

    fp_dir = base / "Fixed Patterns"
    if fp_dir.is_dir():
        for p in sorted(fp_dir.iterdir()):
            if not p.is_dir():
                continue
            gt_candidates = sorted((p / "GT").glob(f"{p.name}_GT.*")) if (p / "GT").is_dir() else []
            mask_candidates = sorted((p / "GT").glob(f"Mask_{p.name}_GT.*")) if (p / "GT").is_dir() else []
            specs.append(
                RealSceneSpec(
                    dataset="otis",
                    scene_id=p.name,
                    category="fixed_pattern",
                    source_type="png_dir",
                    source_path=p,
                    gt_path=gt_candidates[0] if gt_candidates else None,
                    mask_path=mask_candidates[0] if mask_candidates else None,
                )
            )

    fb_dir = base / "Fixed Backgrounds"
    if fb_dir.is_dir():
        for p in sorted(fb_dir.iterdir()):
            if p.is_dir():
                specs.append(
                    RealSceneSpec(
                        dataset="otis",
                        scene_id=p.name,
                        category="real_static",
                        source_type="png_dir",
                        source_path=p,
                    )
                )

    if include_moving and (base / "Moving Target").is_dir():
        for p in sorted((base / "Moving Target").iterdir()):
            if not p.is_dir():
                continue
            specs.append(
                RealSceneSpec(
                    dataset="otis",
                    scene_id=p.name,
                    category="real_dynamic",
                    source_type="png_dir",
                    source_path=p,
                )
            )
    return specs


def _discover_bvi_clear(root: Path) -> List[RealSceneSpec]:
    base = root / "Real_datasets" if (root / "Real_datasets").is_dir() else root
    specs: List[RealSceneSpec] = []
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        pngs = sorted(p.rglob("*.png"))
        if not pngs:
            continue
        tifs = sorted(p.glob("*.tif"))
        frame_dir = pngs[0].parent
        cat = "real_dynamic" if "moving" in p.name.lower() else "real_static"
        specs.append(
            RealSceneSpec(
                dataset="bvi_clear",
                scene_id=p.name,
                category=cat,
                source_type="png_dir",
                source_path=frame_dir,
                clear_ref_path=tifs[0] if tifs else None,
            )
        )
    return specs


def _discover_rlr_at(root: Path) -> List[RealSceneSpec]:
    base = root / "RLR-AT" if (root / "RLR-AT").is_dir() else root
    videos = sorted(p for p in base.iterdir() if p.is_file() and p.suffix.lower() == ".mp4")
    specs: List[RealSceneSpec] = []
    for vid in videos:
        lower = vid.stem.lower()
        cat = "real_dynamic" if ("moving" in lower or "tricycle" in lower) else "real_static"
        specs.append(
            RealSceneSpec(
                dataset="rlr_at",
                scene_id=_sanitize_id(vid.stem),
                category=cat,
                source_type="mp4",
                source_path=vid,
            )
        )
    return specs


def _resolve_ffmpeg_bins() -> Tuple[Optional[str], Optional[str]]:
    ffmpeg_bin = shutil.which("ffmpeg")
    ffprobe_bin = shutil.which("ffprobe")
    if ffmpeg_bin is None:
        try:
            import imageio_ffmpeg  # type: ignore

            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_bin = None
    return ffmpeg_bin, ffprobe_bin


def _probe_video(video_path: Path, ffmpeg_bin: str, ffprobe_bin: Optional[str]) -> Tuple[int, int, float, int]:
    """Return `(width, height, fps, total_frames)` using ffprobe or ffmpeg stderr."""
    if ffprobe_bin:
        cmd = [
            ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames",
            "-of",
            "json",
            str(video_path),
        ]
        out = subprocess.check_output(cmd, text=True)
        stream = json.loads(out)["streams"][0]
        w = int(stream["width"])
        h = int(stream["height"])
        num, den = stream["r_frame_rate"].split("/")
        fps = float(num) / max(float(den), 1.0)
        nb_frames = int(stream.get("nb_frames") or 0)
        if nb_frames <= 0:
            cnt_cmd = [
                ffprobe_bin,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_packets",
                "-show_entries",
                "stream=nb_read_packets",
                "-of",
                "csv=p=0",
                str(video_path),
            ]
            nb_frames = int(subprocess.check_output(cnt_cmd, text=True).strip())
        return w, h, fps, nb_frames

    proc = subprocess.run([ffmpeg_bin, "-i", str(video_path)], capture_output=True, text=True)
    info = proc.stderr or ""
    dur_m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", info)
    size_m = re.search(r"Stream\s+#\d+:\d+.*?Video:.*?,\s*(\d{2,5})x(\d{2,5})", info)
    fps_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:fps|tbr)", info)
    if not size_m:
        raise RuntimeError(f"Could not probe video dimensions via {ffmpeg_bin} for {video_path}")
    w, h = int(size_m.group(1)), int(size_m.group(2))
    fps = float(fps_m.group(1)) if fps_m else 30.0
    if dur_m:
        dur_s = int(dur_m.group(1)) * 3600 + int(dur_m.group(2)) * 60 + float(dur_m.group(3))
        total = max(1, int(round(dur_s * fps)))
    else:
        total = 1
    return w, h, fps, total


def _load_mp4_center_burst(video_path: Path, num_frames: int, stride: int = 1) -> torch.Tensor:
    """Decode a consecutive center burst `[T, 3, H, W]` in `[0, 1]` via ffmpeg, imageio-ffmpeg, PyAV, cv2, or torchvision."""
    ffmpeg_bin, ffprobe_bin = _resolve_ffmpeg_bins()
    if ffmpeg_bin:
        w, h, fps, total = _probe_video(video_path, ffmpeg_bin=ffmpeg_bin, ffprobe_bin=ffprobe_bin)
        span = (num_frames - 1) * max(stride, 1) + 1
        start_frame = max(0, (total - span) // 2)
        start_sec = start_frame / max(fps, 1.0)
        cmd = [
            ffmpeg_bin,
            "-v",
            "error",
            "-ss",
            f"{start_sec:.4f}",
            "-i",
            str(video_path),
            "-vf",
            f"select=not(mod(n\\,{max(stride, 1)}))" if stride > 1 else "null",
            "-frames:v",
            str(num_frames),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
        raw = subprocess.check_output(cmd)
        frame_bytes = w * h * 3
        decoded = len(raw) // frame_bytes
        if decoded == 0:
            raise RuntimeError(f"ffmpeg decoded 0 frames from {video_path}")
        arr = np.frombuffer(raw[: decoded * frame_bytes], dtype=np.uint8).reshape(decoded, h, w, 3).copy()
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().float() / 255.0
        if tensor.shape[0] < num_frames:
            idx = np.linspace(0, tensor.shape[0] - 1, num_frames, dtype=int)
            tensor = tensor[idx]
        return tensor

    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            step = max(stride, 1)
            span = (num_frames - 1) * step + 1
            start_frame = max(0, (total - span) // 2) if total > 0 else 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            collected_cv: List[np.ndarray] = []
            for _ in range(num_frames):
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                collected_cv.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                for _ in range(step - 1):
                    if not cap.grab():
                        break
            cap.release()
            if collected_cv:
                arr = np.stack(collected_cv, axis=0)
                tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().float() / 255.0
                if tensor.shape[0] < num_frames:
                    idx = np.linspace(0, tensor.shape[0] - 1, num_frames, dtype=int)
                    tensor = tensor[idx]
                return tensor
        cap.release()
    except ImportError:
        pass

    try:
        import av  # type: ignore

        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            total = int(stream.frames or 0)
            if total <= 0 and stream.duration and stream.time_base and stream.average_rate:
                total = int(float(stream.duration * stream.time_base) * float(stream.average_rate))
            step = max(stride, 1)
            span = (num_frames - 1) * step + 1
            start_frame = max(0, (total - span) // 2) if total > 0 else 0
            target_indices = {start_frame + i * step for i in range(num_frames)}
            max_idx = max(target_indices)

            if start_frame > 0 and stream.average_rate and stream.time_base:
                approx_sec = max(0.0, (start_frame - 30) / float(stream.average_rate))
                container.seek(int(approx_sec / float(stream.time_base)), stream=stream, backward=True, any_frame=False)

            collected: List[np.ndarray] = []
            fps = float(stream.average_rate) if stream.average_rate else 30.0
            for idx_fallback, frame in enumerate(container.decode(video=0)):
                if frame.pts is not None and stream.time_base is not None:
                    f_idx = int(round(float(frame.pts * stream.time_base) * fps))
                else:
                    f_idx = idx_fallback
                if f_idx in target_indices:
                    collected.append(frame.to_ndarray(format="rgb24"))
                    if len(collected) >= num_frames:
                        break
                elif f_idx > max_idx and collected:
                    break

            if not collected:
                container.seek(0)
                for i, frame in enumerate(container.decode(video=0)):
                    if i >= start_frame and (i - start_frame) % step == 0:
                        collected.append(frame.to_ndarray(format="rgb24"))
                        if len(collected) >= num_frames:
                            break

            if not collected:
                raise RuntimeError(f"PyAV decoded 0 frames from {video_path}")
            arr = np.stack(collected, axis=0)
            tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().float() / 255.0
            if tensor.shape[0] < num_frames:
                idx = np.linspace(0, tensor.shape[0] - 1, num_frames, dtype=int)
                tensor = tensor[idx]
            return tensor
    except ImportError:
        pass

    import torchvision.io as tio

    video, _, _ = tio.read_video(str(video_path), pts_unit="sec", output_format="TCHW")
    total = int(video.shape[0])
    span = (num_frames - 1) * max(stride, 1) + 1
    start = max(0, (total - span) // 2)
    indices = list(range(start, min(total, start + span), max(stride, 1)))[:num_frames]
    while len(indices) < num_frames:
        indices.append(indices[-1])
    return video[indices].float() / 255.0


def _load_png_dir_indices(png_dir: Path) -> List[Path]:
    paths = sorted(p for p in png_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png")
    if not paths:
        raise RuntimeError(f"No PNG frames found in {png_dir}")
    return paths


def _load_image_tensor(path: Path) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.uint8, copy=True)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().float() / 255.0


def compute_sharpness_metrics(img: torch.Tensor) -> Tuple[float, float]:
    """Return `(tenengrad, laplacian_var)` on `[1, 3, H, W]` in `[0, 1]`."""
    x = img.detach().float().clamp(0.0, 1.0)
    luma = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=x.device).view(1, 1, 3, 3)
    ky = kx.transpose(-2, -1)
    gx = F.conv2d(luma, kx, padding=1)
    gy = F.conv2d(luma, ky, padding=1)
    tenengrad = float((gx * gx + gy * gy).mean().item() * 1000.0)

    klap = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=x.device).view(1, 1, 3, 3)
    lap = F.conv2d(luma, klap, padding=1)
    lap_var = float(lap.var(unbiased=False).item() * 1000.0)
    return tenengrad, lap_var


def compute_masked_psnr(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float:
    p = pred.detach().float().clamp(0.0, 1.0)
    g = gt.detach().float().clamp(0.0, 1.0)
    m = (mask > 0.5).float().expand_as(p)
    denom = m.sum().clamp_min(1.0)
    mse = (((p - g) ** 2) * m).sum() / denom
    return float((-10.0 * torch.log10(mse.clamp_min(1.0e-12))).item())


def tensor_to_pil(img: torch.Tensor) -> Image.Image:
    arr = img.detach().cpu().float().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0
    return Image.fromarray(arr.round().astype(np.uint8))


def _label_panel(pil_img: Image.Image, label: str, max_width: int) -> Image.Image:
    w, h = pil_img.size
    if max_width > 0 and w > max_width:
        new_h = max(1, int(round(h * (max_width / float(w)))))
        pil_img = pil_img.resize((max_width, new_h), Image.BILINEAR)
        w, h = pil_img.size

    banner_h = 22
    canvas = Image.new("RGB", (w, h + banner_h), color=(20, 20, 20))
    canvas.paste(pil_img, (0, banner_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), label, fill=(245, 245, 245))
    return canvas


def save_comparison_strip(panels: Sequence[Tuple[str, Image.Image]], out_path: Path, max_width: int) -> None:
    labeled = [_label_panel(img, lbl, max_width=max_width) for lbl, img in panels]
    gap = 4
    total_w = sum(im.width for im in labeled) + gap * max(0, len(labeled) - 1)
    max_h = max(im.height for im in labeled)
    strip = Image.new("RGB", (total_w, max_h), color=(40, 40, 40))
    x = 0
    for im in labeled:
        strip.paste(im, (x, 0))
        x += im.width + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    strip.save(out_path)


def _find_best_detail_box(ref_img: torch.Tensor, crop_size: int = 512) -> Tuple[int, int, int, int]:
    """Locate highest-gradient `crop_size x crop_size` window for 1:1 inspection on 1080p frames."""
    _, _, h, w = ref_img.shape
    cs = min(crop_size, h, w)
    luma = (0.299 * ref_img[:, 0:1] + 0.587 * ref_img[:, 1:2] + 0.114 * ref_img[:, 2:3]).float()
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=ref_img.device).view(1, 1, 3, 3)
    ky = kx.transpose(-2, -1)
    grad = F.conv2d(luma, kx, padding=1).abs() + F.conv2d(luma, ky, padding=1).abs()
    pooled = F.avg_pool2d(grad, kernel_size=cs, stride=max(32, cs // 4))
    flat_idx = int(pooled.view(-1).argmax().item())
    ph, pw = int(pooled.shape[-2]), int(pooled.shape[-1])
    gy = flat_idx // pw
    gx = flat_idx % pw
    stride = max(32, cs // 4)
    y1 = min(gy * stride, h - cs)
    x1 = min(gx * stride, w - cs)
    return x1, y1, x1 + cs, y1 + cs


def _load_neural_checkpoints(
    ckpt_args: Sequence[str], also_stabilize: bool, device: torch.device
) -> List[LoadedModel]:
    loaded: List[LoadedModel] = []
    for item in ckpt_args:
        if "=" not in item:
            raise ValueError(f"Expected `label=/path/to/ckpt.pt`, got: {item}")
        label, path_str = item.split("=", 1)
        ckpt_path = Path(path_str)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found for `{label}`: {ckpt_path}")

        try:
            peek = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            peek = torch.load(ckpt_path, map_location="cpu")
        cfg = validate_config(copy.deepcopy(peek["config"]))
        model = build_model(cfg).to(device)
        AtomicCheckpointer(ckpt_path.parent).load(ckpt_path, model=model, map_location=str(device))
        model.eval()
        nf = int(cfg["model"]["num_frames"])
        _, amp_dtype, _ = resolve_amp_config(device, str(cfg["optim"].get("amp_dtype", "float32")).lower())
        loaded.append(LoadedModel(name=label, model=model, num_frames=nf, amp_dtype=amp_dtype, stabilize=False))
        if also_stabilize:
            loaded.append(
                LoadedModel(name=f"{label}_stab", model=model, num_frames=nf, amp_dtype=amp_dtype, stabilize=True)
            )
    return loaded


def _load_subset_frames(frame_paths: Sequence[Path], indices: Sequence[int]) -> Tuple[torch.Tensor, Dict[int, int]]:
    unique_idx = sorted(set(int(i) for i in indices))
    arrs = [np.asarray(Image.open(frame_paths[i]).convert("RGB"), dtype=np.uint8) for i in unique_idx]
    tensor = torch.from_numpy(np.stack(arrs, axis=0)).permute(0, 3, 1, 2).contiguous().float() / 255.0
    pos_map = {orig_i: local_i for local_i, orig_i in enumerate(unique_idx)}
    return tensor, pos_map


def evaluate_scene(
    spec: RealSceneSpec,
    models: Sequence[LoadedModel],
    classical_frames: int,
    num_windows: int,
    rlr_stride: int,
    anchor_lucky: torch.nn.Module,
    stabilizer: torch.nn.Module,
    tile_size: int,
    tile_overlap: int,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Run all baselines + neural checkpoints on one scene; return `(predictions_by_method, gt, mask)`."""
    preds: Dict[str, torch.Tensor] = {}

    # Determine required frame windows per requested length
    needed_lengths = {classical_frames} | {m.num_frames for m in models}
    base_eval = TiledInference(tile_size=tile_size, overlap=tile_overlap, max_eval_windows=num_windows)

    if spec.source_type == "mp4":
        max_nf = max(needed_lengths)
        center_seq = _load_mp4_center_burst(spec.source_path, num_frames=max_nf, stride=rlr_stride).to(device)
        preds["raw_center"] = center_seq[max_nf // 2 : max_nf // 2 + 1].float()

        def get_windows(nf: int) -> List[torch.Tensor]:
            start = (max_nf - nf) // 2
            return [center_seq[start : start + nf]]
    else:
        frame_paths = _load_png_dir_indices(spec.source_path)
        total = len(frame_paths)
        center_idx = total // 2
        all_windows: Dict[int, List[List[int]]] = {
            nf: base_eval.select_eval_windows(total, nf) for nf in needed_lengths
        }
        union_indices = [center_idx] + [i for wins in all_windows.values() for w in wins for i in w]
        loaded_seq, pos_map = _load_subset_frames(frame_paths, union_indices)
        loaded_seq = loaded_seq.to(device)
        preds["raw_center"] = loaded_seq[pos_map[center_idx] : pos_map[center_idx] + 1].float()

        def get_windows(nf: int) -> List[torch.Tensor]:
            return [loaded_seq[[pos_map[i] for i in w]] for w in all_windows[nf]]

    # Classical baselines (`raw_mean`, `A0-lucky`, optional `A0-lucky_stab`)
    c_wins = get_windows(classical_frames)
    preds["raw_mean"] = torch.stack([w.mean(dim=0) for w in c_wins], dim=0).mean(dim=0, keepdim=True).float()
    with torch.inference_mode():
        lucky_outs = [anchor_lucky.build(w.unsqueeze(0)).float() for w in c_wins]
    preds["A0-lucky"] = torch.stack(lucky_outs, dim=0).mean(dim=0)

    if spec.clear_ref_path is not None and spec.clear_ref_path.is_file():
        preds["CLEAR_official"] = _load_image_tensor(spec.clear_ref_path).unsqueeze(0).to(device)

    # Neural models
    for lm in models:
        evaluator = TiledInference(
            tile_size=tile_size,
            overlap=tile_overlap,
            max_eval_windows=num_windows,
            stabilize_in_eval=lm.stabilize,
        )
        m_wins = get_windows(lm.num_frames)
        outs: List[torch.Tensor] = []
        for w in m_wins:
            pred = evaluator.predict_burst(lm.model, w.unsqueeze(0), amp_dtype=lm.amp_dtype)
            outs.append(pred.restored.float())
        preds[lm.name] = torch.stack(outs, dim=0).mean(dim=0)

    gt = _load_image_tensor(spec.gt_path).unsqueeze(0).to(device) if spec.gt_path else None
    mask: Optional[torch.Tensor] = None
    if spec.mask_path is not None and spec.mask_path.is_file():
        m_arr = np.asarray(Image.open(spec.mask_path).convert("L"), dtype=np.float32) / 255.0
        mask = torch.from_numpy(m_arr).unsqueeze(0).unsqueeze(0).to(device)

    del stabilizer  # kept symmetrical if classical_stab is added later
    return preds, gt, mask


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    specs: List[RealSceneSpec] = []
    if args.otis_root is not None:
        otis_specs = _discover_otis(args.otis_root, include_moving=args.include_otis_moving)
        specs.extend(otis_specs[: args.max_scenes_per_dataset] if args.max_scenes_per_dataset > 0 else otis_specs)
    if args.bvi_root is not None:
        bvi_specs = _discover_bvi_clear(args.bvi_root)
        specs.extend(bvi_specs[: args.max_scenes_per_dataset] if args.max_scenes_per_dataset > 0 else bvi_specs)
    if args.rlr_root is not None:
        rlr_specs = _discover_rlr_at(args.rlr_root)
        specs.extend(rlr_specs[: args.max_scenes_per_dataset] if args.max_scenes_per_dataset > 0 else rlr_specs)

    if not specs:
        raise ValueError("No datasets discovered. Pass at least one of --otis-root, --bvi-root, or --rlr-root.")

    base_cfg = validate_config(load_config(args.config, variant=None))
    anchor_lucky = build_anchor_builder(base_cfg["model"], anchor_override="lucky").to(device)
    stabilizer = STABILIZER_REGISTRY.build("fft_global").to(device)
    tile_size = int(base_cfg["eval"].get("tile_size", 256))
    tile_overlap = int(base_cfg["eval"].get("tile_overlap", 64))

    models = _load_neural_checkpoints(args.checkpoints, also_stabilize=args.also_stabilize, device=device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for idx, spec in enumerate(specs, 1):
        preds, gt, mask = evaluate_scene(
            spec=spec,
            models=models,
            classical_frames=args.classical_frames,
            num_windows=args.num_windows,
            rlr_stride=args.rlr_fps_stride,
            anchor_lucky=anchor_lucky,
            stabilizer=stabilizer,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            device=device,
        )

        raw_ten, raw_lap = compute_sharpness_metrics(preds["raw_center"])
        raw_psnr = compute_psnr(preds["raw_center"], gt) if gt is not None else None

        strip_panels: List[Tuple[str, Image.Image]] = []
        pil_by_method: Dict[str, Image.Image] = {}

        for method_name, pred_img in preds.items():
            ten, lap = compute_sharpness_metrics(pred_img)
            psnr = compute_psnr(pred_img, gt) if gt is not None else None
            ssim = compute_ssim(pred_img, gt) if gt is not None else None
            cw_ssim = compute_cw_ssim(pred_img, gt) if gt is not None else None
            masked_psnr = compute_masked_psnr(pred_img, gt, mask) if (gt is not None and mask is not None) else None

            row = {
                "dataset": spec.dataset,
                "category": spec.category,
                "scene_id": spec.scene_id,
                "method": method_name,
                "has_gt": gt is not None,
                "psnr": psnr,
                "masked_psnr": masked_psnr,
                "ssim": ssim,
                "cw_ssim": cw_ssim,
                "psnr_gain_vs_raw": (psnr - raw_psnr) if (psnr is not None and raw_psnr is not None) else None,
                "tenengrad": ten,
                "laplacian_var": lap,
                "tenengrad_ratio_vs_raw": ten / max(raw_ten, 1.0e-8),
                "laplacian_ratio_vs_raw": lap / max(raw_lap, 1.0e-8),
            }
            rows.append(row)

            pil_im = tensor_to_pil(pred_img[0])
            pil_by_method[method_name] = pil_im
            out_scene = args.output_dir / "scenes" / spec.dataset / method_name / f"{spec.scene_id}.png"
            out_scene.parent.mkdir(parents=True, exist_ok=True)
            pil_im.save(out_scene)

            short_lbl = f"{method_name} ({psnr:.2f}dB)" if psnr is not None else f"{method_name} (T:{ten:.1f})"
            strip_panels.append((short_lbl, pil_im))

        if gt is not None:
            gt_pil = tensor_to_pil(gt[0])
            gt_out = args.output_dir / "scenes" / spec.dataset / "gt" / f"{spec.scene_id}.png"
            gt_out.parent.mkdir(parents=True, exist_ok=True)
            gt_pil.save(gt_out)
            strip_panels.append(("GT", gt_pil))

        save_comparison_strip(
            strip_panels,
            args.output_dir / "comparisons" / spec.dataset / f"{spec.scene_id}_strip.png",
            max_width=args.strip_max_width,
        )

        # Save native 512x512 high-detail crop strip for large frames (e.g. 1080p RLR-AT)
        if preds["raw_center"].shape[-1] > 768 or preds["raw_center"].shape[-2] > 768:
            ref_tensor = preds[models[-1].name] if models else preds["A0-lucky"]
            box = _find_best_detail_box(ref_tensor, crop_size=512)
            crop_panels = [(lbl, im.crop(box)) for lbl, im in strip_panels]
            save_comparison_strip(
                crop_panels,
                args.output_dir / "comparisons" / spec.dataset / f"{spec.scene_id}_crop512.png",
                max_width=0,
            )

        print(f"[{idx:3d}/{len(specs)}] {spec.dataset}/{spec.scene_id} ({spec.category}) concluido.")

    # Save CSV
    csv_path = args.output_dir / "metrics_per_scene.csv"
    fieldnames = [
        "dataset",
        "category",
        "scene_id",
        "method",
        "has_gt",
        "psnr",
        "masked_psnr",
        "ssim",
        "cw_ssim",
        "psnr_gain_vs_raw",
        "tenengrad",
        "laplacian_var",
        "tenengrad_ratio_vs_raw",
        "laplacian_ratio_vs_raw",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate summary by (dataset, category, method) and rank top visual scenes
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault((r["dataset"], r["category"], r["method"]), []).append(r)

    aggregates: List[Dict[str, Any]] = []
    for (ds, cat, meth), items in sorted(grouped.items()):
        entry: Dict[str, Any] = {"dataset": ds, "category": cat, "method": meth, "num_scenes": len(items)}
        if items[0]["has_gt"]:
            entry["mean_psnr"] = float(np.mean([x["psnr"] for x in items]))
            entry["mean_masked_psnr"] = float(np.mean([x["masked_psnr"] for x in items if x["masked_psnr"] is not None])) if any(x["masked_psnr"] is not None for x in items) else None
            entry["mean_ssim"] = float(np.mean([x["ssim"] for x in items]))
            entry["mean_cw_ssim"] = float(np.mean([x["cw_ssim"] for x in items]))
            entry["mean_psnr_gain_vs_raw"] = float(np.mean([x["psnr_gain_vs_raw"] for x in items]))
        entry["mean_tenengrad"] = float(np.mean([x["tenengrad"] for x in items]))
        entry["mean_laplacian_var"] = float(np.mean([x["laplacian_var"] for x in items]))
        aggregates.append(entry)

    # Top scenes for easy visual picking (sorted by PSNR gain if GT else Tenengrad ratio on last neural model or A0-lucky)
    target_method = models[-1].name if models else "A0-lucky"
    top_rows = [r for r in rows if r["method"] == target_method]
    top_rows.sort(
        key=lambda r: (r["psnr_gain_vs_raw"] if r["psnr_gain_vs_raw"] is not None else r["tenengrad_ratio_vs_raw"]),
        reverse=True,
    )

    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "aggregates": aggregates,
                "top_scenes_for_report": top_rows[:25],
                "per_scene": rows,
            },
            handle,
            indent=2,
        )

    print("\n=== Resumo Datasets Reais ===")
    for a in aggregates:
        if "mean_psnr" in a:
            m_str = f" | MaskPSNR={a['mean_masked_psnr']:.2f}" if a.get("mean_masked_psnr") is not None else ""
            print(
                f"{a['dataset']:10s} | {a['category']:13s} | {a['method']:18s} (n={a['num_scenes']:2d}) | "
                f"PSNR={a['mean_psnr']:.2f}{m_str} | SSIM={a['mean_ssim']:.4f} | CW-SSIM={a['mean_cw_ssim']:.4f}"
            )
        else:
            print(
                f"{a['dataset']:10s} | {a['category']:13s} | {a['method']:18s} (n={a['num_scenes']:2d}) | "
                f"Tenengrad={a['mean_tenengrad']:.2f} | LapVar={a['mean_laplacian_var']:.2f}"
            )
    print(f"\nCSV salvo em: {csv_path}")
    print(f"JSON salvo em: {summary_path}")


if __name__ == "__main__":
    main()
