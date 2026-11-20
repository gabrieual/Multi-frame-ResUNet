"""Evaluate restored Turbulence-Text crops using pretrained CRNN, DAN, and ASTER models."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import re
import string
import sys
import types
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

RESUNET_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CRNN, DAN, and ASTER OCR evaluation on Turbulence-Text manifests."
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=RESUNET_ROOT / "eval_vis" / "turb_text",
        help="Root directory containing <method>/manifest.json folders.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional list of method folder names under --eval-root (defaults to all found).",
    )
    parser.add_argument(
        "--manifest-name",
        type=str,
        default="manifest.json",
        help="Manifest filename inside each method dir (`manifest.json` or `manifest_per_window.json`).",
    )
    parser.add_argument(
        "--ocr-tools-dir",
        type=Path,
        default=Path.home() / "ocr_tools",
        help="Directory populated by setup_turb_text_ocr.sh.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["crnn", "dan", "aster"],
        choices=["crnn", "dan", "aster"],
        help="Subset of OCR recognizers to run.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional limit on crops per method (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=RESUNET_ROOT / "eval_reports" / "turb_text_ocr.json",
        help="Output JSON report path.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """Standard ICDAR / UG2+ case-insensitive alphanumeric normalization."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def score_predictions(preds: Sequence[str], targets: Sequence[str]) -> Dict[str, float]:
    assert len(preds) == len(targets)
    if not targets:
        return {"accuracy": 0.0, "ned": 0.0, "num_samples": 0}
    correct = 0
    ned_sum = 0.0
    for pred_raw, gt_raw in zip(preds, targets):
        p = normalize_text(pred_raw)
        g = normalize_text(gt_raw)
        if p == g:
            correct += 1
        max_len = max(len(p), len(g), 1)
        ned_sum += 1.0 - (levenshtein_distance(p, g) / float(max_len))
    n = float(len(targets))
    return {
        "accuracy": 100.0 * correct / n,
        "ned": 100.0 * ned_sum / n,
        "correct": correct,
        "num_samples": len(targets),
    }


def _torch_load_compat(path: Path, map_location: torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in state_dict):
        return {k.removeprefix("module."): v for k, v in state_dict.items()}
    return state_dict


# ---------------------------------------------------------------------------
# 1. CRNN Adapter (clovaai/deep-text-recognition-benchmark)
# ---------------------------------------------------------------------------
class CRNNRecognizer:
    def __init__(self, repo_dir: Path, ckpt_path: Path, device: torch.device) -> None:
        if not repo_dir.is_dir():
            raise FileNotFoundError(f"CRNN repo not found: {repo_dir}")
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"CRNN checkpoint not found: {ckpt_path}")

        self.device = device
        self.character = "0123456789abcdefghijklmnopqrstuvwxyz"
        self.dict = {ch: i + 1 for i, ch in enumerate(self.character)}  # 0 is CTC blank
        self.chars = ["[blank]"] + list(self.character)

        @dataclass
        class _CRNNOpt:
            Transformation: str = "None"
            FeatureExtraction: str = "VGG"
            SequenceModeling: str = "BiLSTM"
            Prediction: str = "CTC"
            num_fiducial: int = 20
            imgH: int = 32
            imgW: int = 100
            input_channel: int = 1
            output_channel: int = 512
            hidden_size: int = 256
            num_class: int = len(self.chars)
            batch_max_length: int = 25

        opt = _CRNNOpt()
        sys.path.insert(0, str(repo_dir))
        try:
            for mod_name in list(sys.modules):
                if mod_name == "model" or mod_name == "modules" or mod_name.startswith("modules."):
                    sys.modules.pop(mod_name, None)
            crnn_mod = importlib.import_module("model")
            self.model = crnn_mod.Model(opt).to(device)
        finally:
            if sys.path and sys.path[0] == str(repo_dir):
                sys.path.pop(0)

        state = _torch_load_compat(ckpt_path, map_location=device)
        self.model.load_state_dict(_strip_module_prefix(state))
        self.model.eval()

    @torch.inference_mode()
    def predict_batch(self, image_paths: Sequence[Path]) -> List[str]:
        tensors = []
        for p in image_paths:
            img = Image.open(p).convert("L").resize((100, 32), Image.BICUBIC)
            arr = (np.asarray(img, dtype=np.float32) / 255.0 - 0.5) / 0.5
            tensors.append(torch.from_numpy(arr).unsqueeze(0))
        batch = torch.stack(tensors, dim=0).to(self.device)
        dummy_text = torch.zeros((batch.shape[0], 26), dtype=torch.long, device=self.device)
        preds = self.model(batch, dummy_text)
        preds_index = preds.argmax(dim=2).cpu().tolist()

        texts: List[str] = []
        for seq in preds_index:
            out_chars: List[str] = []
            prev = 0
            for idx in seq:
                if idx != 0 and idx != prev:
                    out_chars.append(self.chars[idx])
                prev = idx
            texts.append("".join(out_chars))
        return texts


# ---------------------------------------------------------------------------
# 2. DAN Adapter (Wang-Tianwei/Decoupled-attention-network)
# ---------------------------------------------------------------------------
class DANRecognizer:
    def __init__(self, repo_dir: Path, weights_dir: Path, device: torch.device) -> None:
        if not repo_dir.is_dir():
            raise FileNotFoundError(f"DAN repo not found: {repo_dir}")
        scene_dir = weights_dir / "scene"
        if not scene_dir.is_dir():
            scene_dir = weights_dir

        m0 = sorted(scene_dir.glob("*_M0.pth"))
        m1 = sorted(scene_dir.glob("*_M1.pth"))
        m2 = sorted(scene_dir.glob("*_M2.pth"))
        if not (m0 and m1 and m2):
            raise FileNotFoundError(f"DAN scene weights (*_M0/M1/M2.pth) not found under {weights_dir}")

        dict_path = repo_dir / "dict" / "dic_36.txt"
        with open(dict_path, "r", encoding="utf-8") as handle:
            self.vocab = [line.rstrip("\r\n") for line in handle]

        self.device = device
        sys.path.insert(0, str(repo_dir))
        try:
            for mod_name in ("DAN", "resnet"):
                sys.modules.pop(mod_name, None)
            dan_mod = importlib.import_module("DAN")
            self.fe = dan_mod.Feature_Extractor(
                strides=[(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)],
                compress_layer=False,
                input_shape=[1, 32, 128],
            )
            scales = self.fe.Iwantshapes()
            self.cam = dan_mod.CAM(scales=scales, maxT=25, depth=8, num_channels=64)
            self.dtd = dan_mod.DTD(nclass=len(self.vocab) + 2, nchannel=512, dropout=0.3)
        finally:
            if sys.path and sys.path[0] == str(repo_dir):
                sys.path.pop(0)

        self.fe.load_state_dict(_strip_module_prefix(_torch_load_compat(m0[0], device)))
        self.cam.load_state_dict(_strip_module_prefix(_torch_load_compat(m1[0], device)))
        self.dtd.load_state_dict(_strip_module_prefix(_torch_load_compat(m2[0], device)))
        self.fe.to(device).eval()
        self.cam.to(device).eval()
        self.dtd.to(device).eval()

    @staticmethod
    def _preprocess_one(path: Path) -> torch.Tensor:
        img = Image.open(path).convert("L")
        w, h = img.size
        cur_ratio = w / float(max(h, 1))
        target_h, target_w = 32, 128
        target_ratio = target_w / float(target_h)
        if cur_ratio > target_ratio:
            resized_w = target_w
        else:
            resized_w = max(1, min(target_w, int(target_h * cur_ratio)))
        resized = np.asarray(img.resize((resized_w, target_h), Image.BILINEAR), dtype=np.uint8)
        canvas = np.zeros((target_h, target_w), dtype=np.uint8)
        start_x = (target_w - resized_w) // 2
        canvas[:, start_x : start_x + resized_w] = resized
        return torch.from_numpy(canvas.astype(np.float32) / 255.0).unsqueeze(0)

    @torch.inference_mode()
    def predict_batch(self, image_paths: Sequence[Path]) -> List[str]:
        orig_len = len(image_paths)
        tensors = [self._preprocess_one(p) for p in image_paths]
        # DTD.forward(..., test=True) uses .squeeze() which breaks on batch_size == 1
        if orig_len == 1:
            tensors.append(tensors[0])
        batch = torch.stack(tensors, dim=0).to(self.device)

        features = self.fe(batch)
        attn_maps = self.cam(features)
        dummy_target = torch.zeros((batch.shape[0], 25), dtype=torch.long, device=self.device)
        dummy_len = torch.full((batch.shape[0],), 25, dtype=torch.long, device=self.device)
        output, out_length = self.dtd(features[-1], attn_maps, dummy_target, dummy_len, True)

        preds_idx = output.argmax(dim=1).cpu().tolist()
        lengths = [int(x) for x in out_length.cpu().tolist()]
        texts: List[str] = []
        offset = 0
        for length in lengths[:orig_len]:
            seq = preds_idx[offset : offset + length]
            offset += length
            chars = [self.vocab[idx - 1] for idx in seq if 0 < idx <= len(self.vocab)]
            texts.append("".join(chars))
        return texts


# ---------------------------------------------------------------------------
# 3. ASTER Adapter (ayumiymk/aster.pytorch)
# ---------------------------------------------------------------------------
class ASTERRecognizer:
    def __init__(self, repo_dir: Path, ckpt_path: Path, device: torch.device) -> None:
        if not repo_dir.is_dir():
            raise FileNotFoundError(f"ASTER repo not found: {repo_dir}")
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"ASTER checkpoint not found: {ckpt_path}")

        self.device = device
        self.voc = list(string.printable[:-6]) + ["EOS", "PADDING", "UNKNOWN"]
        self.eos_id = self.voc.index("EOS")
        self.unknown_id = self.voc.index("UNKNOWN")

        aster_cfg = types.SimpleNamespace(
            height=64,
            width=256,
            voc_type="ALLCASES_SYMBOLS",
            arch="ResNet_ASTER",
            with_lstm=True,
            max_len=100,
            STN_ON=True,
            beam_width=5,
            tps_inputsize=[32, 64],
            tps_outputsize=[32, 100],
            tps_margins=[0.05, 0.05],
            stn_activation="none",
            num_control_points=20,
            n_group=1,
            decoder_sdim=512,
            attDim=512,
        )

        # Stub `lib` and `config` so importing `lib.models.model_builder` skips `lib/__init__.py`
        # (tensorflow/scipy/editdistance) and `config.py` (`import six` + `sys.argv` parsing).
        saved_lib = {
            k: v
            for k, v in sys.modules.items()
            if k == "lib" or k.startswith("lib.") or k == "config"
        }
        for k in list(saved_lib):
            sys.modules.pop(k, None)
        sys.path.insert(0, str(repo_dir))
        try:
            lib_pkg = types.ModuleType("lib")
            lib_pkg.__path__ = [str(repo_dir / "lib")]  # type: ignore[attr-defined]
            sys.modules["lib"] = lib_pkg
            cfg_mod = types.ModuleType("config")
            cfg_mod.get_args = lambda _argv=None: aster_cfg  # type: ignore[attr-defined]
            sys.modules["config"] = cfg_mod
            mb_mod = importlib.import_module("lib.models.model_builder")
            self.model = mb_mod.ModelBuilder(
                arch="ResNet_ASTER",
                rec_num_classes=len(self.voc),
                sDim=512,
                attDim=512,
                max_len_labels=100,
                eos=self.eos_id,
                STN_ON=True,
            ).to(device)
        finally:
            if sys.path and sys.path[0] == str(repo_dir):
                sys.path.pop(0)
            for k in list(sys.modules):
                if k == "lib" or k.startswith("lib.") or k == "config":
                    sys.modules.pop(k, None)
            sys.modules.update(saved_lib)

        ckpt = _torch_load_compat(ckpt_path, map_location=device)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        self.model.load_state_dict(_strip_module_prefix(state))
        self.model.eval()
        self.beam_width = 5

    def _beam_search(self, encoder_feats: torch.Tensor) -> torch.Tensor:
        """PyTorch 2.x / CPU-or-CUDA safe reimplementation of ASTER AttentionRecognitionHead.beam_search."""
        decoder = self.model.decoder
        device = encoder_feats.device
        batch_size, seq_len, feat_dim = encoder_feats.size()
        k = self.beam_width

        inflated_feats = (
            encoder_feats.unsqueeze(1)
            .repeat(1, k, 1, 1)
            .contiguous()
            .view(batch_size * k, seq_len, feat_dim)
        )
        state = torch.zeros(1, batch_size * k, decoder.sDim, device=device)
        pos_index = (torch.arange(batch_size, device=device, dtype=torch.long) * k).view(-1, 1)

        sequence_scores = torch.full((batch_size * k, 1), -float("inf"), device=device)
        start_rows = torch.arange(batch_size, device=device, dtype=torch.long) * k
        sequence_scores.index_fill_(0, start_rows, 0.0)

        y_prev = torch.full((batch_size * k,), decoder.num_classes, dtype=torch.long, device=device)
        stored_scores: List[torch.Tensor] = []
        stored_predecessors: List[torch.Tensor] = []
        stored_symbols: List[torch.Tensor] = []

        for _ in range(decoder.max_len_labels):
            output, state = decoder.decoder(inflated_feats, state, y_prev)
            log_probs = F.log_softmax(output, dim=1)
            scores_expanded = sequence_scores.expand(-1, decoder.num_classes) + log_probs
            scores, candidates = scores_expanded.view(batch_size, -1).topk(k, dim=1)

            y_prev = (candidates % decoder.num_classes).view(batch_size * k)
            sequence_scores = scores.view(batch_size * k, 1)
            predecessors = (torch.div(candidates, decoder.num_classes, rounding_mode="floor") + pos_index.expand_as(candidates)).view(batch_size * k, 1)
            state = state.index_select(1, predecessors.squeeze(1))

            stored_scores.append(sequence_scores.clone())
            eos_mask = y_prev.view(-1, 1).eq(self.eos_id)
            if eos_mask.any():
                sequence_scores.masked_fill_(eos_mask, -float("inf"))

            stored_predecessors.append(predecessors)
            stored_symbols.append(y_prev)

        sorted_score, sorted_idx = stored_scores[-1].view(batch_size, k).topk(k)
        s = sorted_score.clone()
        batch_eos_found = [0] * batch_size
        t_predecessors = (sorted_idx + pos_index.expand_as(sorted_idx)).view(batch_size * k)
        reversed_paths: List[torch.Tensor] = []

        for t in range(decoder.max_len_labels - 1, -1, -1):
            curr_sym = stored_symbols[t].index_select(0, t_predecessors)
            t_predecessors = stored_predecessors[t].index_select(0, t_predecessors).squeeze(1)
            eos_indices = stored_symbols[t].eq(self.eos_id).nonzero(as_tuple=False)
            if eos_indices.numel() > 0:
                for i in range(eos_indices.size(0) - 1, -1, -1):
                    flat_idx = int(eos_indices[i, 0].item())
                    b_idx = flat_idx // k
                    res_k_idx = k - (batch_eos_found[b_idx] % k) - 1
                    batch_eos_found[b_idx] += 1
                    res_idx = b_idx * k + res_k_idx
                    t_predecessors[res_idx] = stored_predecessors[t][flat_idx, 0]
                    curr_sym[res_idx] = stored_symbols[t][flat_idx]
                    s[b_idx, res_k_idx] = stored_scores[t][flat_idx, 0]
            reversed_paths.append(curr_sym)

        _, re_sorted_idx = s.topk(k)
        re_sorted_idx = (re_sorted_idx + pos_index.expand_as(re_sorted_idx)).view(batch_size * k)
        paths = [step.index_select(0, re_sorted_idx).view(batch_size, k, -1) for step in reversed(reversed_paths)]
        return torch.cat(paths, dim=-1)[:, 0, :]

    @torch.inference_mode()
    def predict_batch(self, image_paths: Sequence[Path]) -> List[str]:
        tensors = []
        for p in image_paths:
            img = Image.open(p).convert("RGB").resize((256, 64), Image.BILINEAR)
            arr = (np.asarray(img, dtype=np.float32) / 255.0 - 0.5) / 0.5
            tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
        x = torch.stack(tensors, dim=0).to(self.device)

        if self.model.STN_ON:
            stn_in = F.interpolate(x, self.model.tps_inputsize, mode="bilinear", align_corners=True)
            _, ctrl_points = self.model.stn_head(stn_in)
            x, _ = self.model.tps(x, ctrl_points)

        encoder_feats = self.model.encoder(x).contiguous()
        pred_ids = self._beam_search(encoder_feats).cpu().tolist()

        texts: List[str] = []
        for seq in pred_ids:
            chars: List[str] = []
            for idx in seq:
                if idx == self.eos_id:
                    break
                if idx != self.unknown_id and 0 <= idx < len(self.voc):
                    chars.append(self.voc[idx])
            texts.append("".join(chars))
        return texts


def build_recognizers(
    model_names: Sequence[str], tools_dir: Path, device: torch.device
) -> Dict[str, Any]:
    repos_dir = tools_dir / "repos"
    weights_dir = tools_dir / "weights"
    built: Dict[str, Any] = {}
    for name in model_names:
        if name == "crnn":
            built[name] = CRNNRecognizer(
                repo_dir=repos_dir / "deep-text-recognition-benchmark",
                ckpt_path=weights_dir / "None-VGG-BiLSTM-CTC.pth",
                device=device,
            )
        elif name == "dan":
            built[name] = DANRecognizer(
                repo_dir=repos_dir / "Decoupled-attention-network",
                weights_dir=weights_dir / "dan",
                device=device,
            )
        elif name == "aster":
            built[name] = ASTERRecognizer(
                repo_dir=repos_dir / "aster.pytorch",
                ckpt_path=weights_dir / "aster_demo.pth.tar",
                device=device,
            )
    return built


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.methods:
        method_dirs = [args.eval_root / m for m in args.methods]
    else:
        method_dirs = sorted(d for d in args.eval_root.iterdir() if d.is_dir() and (d / args.manifest_name).is_file())

    if not method_dirs:
        raise FileNotFoundError(f"No method directories with {args.manifest_name} found under {args.eval_root}")

    recognizers = build_recognizers(args.models, args.ocr_tools_dir, device=device)
    summary: Dict[str, Any] = {"device": str(device), "models": list(args.models), "results": {}}

    for method_dir in method_dirs:
        manifest_path = method_dir / args.manifest_name
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        crops = manifest["crops"]
        if args.max_samples > 0:
            crops = crops[: args.max_samples]

        crop_paths = [Path(item["crop_path"]) for item in crops]
        targets = [str(item["text"]) for item in crops]
        method_key = str(manifest.get("method_name", method_dir.name))

        method_metrics: Dict[str, Any] = {"num_samples": len(crop_paths), "per_model": {}}
        for model_name, recognizer in recognizers.items():
            preds: List[str] = []
            for start in range(0, len(crop_paths), args.batch_size):
                preds.extend(recognizer.predict_batch(crop_paths[start : start + args.batch_size]))
            scores = score_predictions(preds, targets)
            method_metrics["per_model"][model_name] = scores
            if args.max_samples > 0:
                print(f"[{method_key} | {model_name.upper()}] sample preds vs gt: {list(zip(preds[:5], targets[:5]))}")

        summary["results"][method_key] = method_metrics

    print("\n=== Resumo Turb-Text OCR (Word Accuracy %: CRNN / DAN / ASTER) ===")
    for method_key, m in summary["results"].items():
        pm = m["per_model"]
        acc_str = " / ".join(
            f"{pm[k]['accuracy']:6.2f}" if k in pm else "  N/A " for k in ("crnn", "dan", "aster")
        )
        ned_str = " / ".join(
            f"{pm[k]['ned']:6.2f}" if k in pm else "  N/A " for k in ("crnn", "dan", "aster")
        )
        print(f"{method_key:24s} (n={m['num_samples']:4d}) | Acc: {acc_str} | 1-NED: {ned_str}")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nRelatorio salvo em: {args.output_json}")


if __name__ == "__main__":
    main()
