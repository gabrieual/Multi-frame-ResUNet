"""Configuration loading, validation, and ablation variant preset resolution."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


# Exact 1-variable-at-a-time ablation presets from ResUNetMultiFrame.md Section 8
ABLATION_PRESETS: Dict[str, Dict[str, Any]] = {
    "A1": {
        "experiment": {"name": "resunet_A1_single_image_mean", "variant": "A1"},
        "model": {"input_mode": "mean_only", "anchor_type": "mean"},
        "loss": {"charbonnier_weight": 1.0, "fft_weight": 0.05, "edge_weight": 0.0, "ssim_weight": 0.0},
    },
    "A2": {
        "experiment": {"name": "resunet_A2_multiframe_only", "variant": "A2"},
        "model": {"input_mode": "frames_only", "anchor_type": "mean"},
        "loss": {"charbonnier_weight": 1.0, "fft_weight": 0.05, "edge_weight": 0.0, "ssim_weight": 0.0},
    },
    "A3a": {
        "experiment": {"name": "resunet_A3a_frames_plus_anchor", "variant": "A3a"},
        "model": {"input_mode": "frames_anchor", "anchor_type": "mean"},
        "loss": {"charbonnier_weight": 1.0, "fft_weight": 0.05, "edge_weight": 0.0, "ssim_weight": 0.0},
    },
    "A3b": {
        "experiment": {"name": "resunet_A3b_frames_anchor_var", "variant": "A3b"},
        "model": {"input_mode": "full", "anchor_type": "mean"},
        "loss": {"charbonnier_weight": 1.0, "fft_weight": 0.05, "edge_weight": 0.0, "ssim_weight": 0.0},
    },
    "A3b-FRF": {
        "experiment": {"name": "resunet_A3b_FRF_anchor", "variant": "A3b-FRF"},
        "model": {"input_mode": "full", "anchor_type": "frf"},
        "loss": {"charbonnier_weight": 1.0, "fft_weight": 0.05, "edge_weight": 0.0, "ssim_weight": 0.0},
    },
    "A3c": {
        "experiment": {"name": "resunet_A3c_full_edge", "variant": "A3c"},
        "model": {"input_mode": "full", "anchor_type": "mean"},
        "loss": {"charbonnier_weight": 1.0, "fft_weight": 0.05, "edge_weight": 0.05, "ssim_weight": 0.0},
    },
    "A3c-FRF": {
        "experiment": {"name": "resunet_A3c_FRF_edge", "variant": "A3c-FRF"},
        "model": {"input_mode": "full", "anchor_type": "frf"},
        "loss": {"charbonnier_weight": 1.0, "fft_weight": 0.05, "edge_weight": 0.05, "ssim_weight": 0.0},
    },
    "A3c-Lucky": {
        "experiment": {"name": "resunet_A3c_Lucky_edge", "variant": "A3c-Lucky"},
        "model": {"input_mode": "full", "anchor_type": "lucky"},
        "loss": {"charbonnier_weight": 1.0, "fft_weight": 0.05, "edge_weight": 0.05, "ssim_weight": 0.0},
    },
    "A3c-Lucky-SSIM": {
        "experiment": {"name": "resunet_A3c_Lucky_edge_ssim", "variant": "A3c-Lucky-SSIM"},
        "model": {"input_mode": "full", "anchor_type": "lucky"},
        "loss": {"charbonnier_weight": 1.0, "fft_weight": 0.05, "edge_weight": 0.05, "ssim_weight": 0.1},
    },
    # Etapa 2 (CLEAR-inspired upgrades over A3c-Lucky-SSIM):
    # 1) Dual FRF + Lucky anchor stack in stem + Complex Wavelet SSIM phase-aware loss
    "A4-FRFLucky-CW": {
        "experiment": {"name": "resunet_A4_frf_lucky_cw", "variant": "A4-FRFLucky-CW"},
        "model": {
            "input_mode": "full_frf_lucky",
            "anchor_type": "lucky",
            "use_pre_align": False,
            "middle_type": "naf",
        },
        "loss": {
            "charbonnier_weight": 1.0,
            "fft_weight": 0.05,
            "edge_weight": 0.05,
            "ssim_weight": 0.1,
            "cw_ssim_weight": 0.1,
        },
    },
    # 2) Two-stage geometric registration (Pre-Stem Flow Aligner -> ResUNet)
    "A4-PreAlign": {
        "experiment": {"name": "resunet_A4_pre_align", "variant": "A4-PreAlign"},
        "model": {
            "input_mode": "full",
            "anchor_type": "lucky",
            "use_pre_align": True,
            "middle_type": "naf",
        },
        "loss": {
            "charbonnier_weight": 1.0,
            "fft_weight": 0.05,
            "edge_weight": 0.05,
            "ssim_weight": 0.1,
            "cw_ssim_weight": 0.0,
        },
    },
    # 3) Global 2D State-Space context (pure-PyTorch SS2D Mamba) in bottleneck middle
    "A4-Mamba": {
        "experiment": {"name": "resunet_A4_mamba_middle", "variant": "A4-Mamba"},
        "model": {
            "input_mode": "full",
            "anchor_type": "lucky",
            "use_pre_align": False,
            "middle_type": "ss2d",
        },
        "loss": {
            "charbonnier_weight": 1.0,
            "fft_weight": 0.05,
            "edge_weight": 0.05,
            "ssim_weight": 0.1,
            "cw_ssim_weight": 0.0,
        },
    },
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges `override` into a copy of `base`."""
    result = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


def apply_variant_preset(cfg: Dict[str, Any], variant: Optional[str]) -> Dict[str, Any]:
    """Applies an ablation preset (A1..A3c) while preserving data/optim overrides."""
    if not variant:
        return cfg
    canonical = None
    for key in ABLATION_PRESETS:
        if key.lower() == variant.strip().lower():
            canonical = key
            break
    if canonical is None:
        valid = ", ".join(ABLATION_PRESETS.keys())
        raise ValueError(f"Unknown ablation variant '{variant}'. Valid: [{valid}]")
    return deep_merge(cfg, ABLATION_PRESETS[canonical])


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validates invariants from ResUNetMultiFrame.md before model/data instantiation."""
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    optim_cfg = cfg["optim"]
    eval_cfg = cfg["eval"]

    if data_cfg.get("allow_synthetic_fallback", False):
        raise ValueError(
            "data.allow_synthetic_fallback MUST be false to prevent silent fake-data training."
        )

    data_n = int(data_cfg["num_frames"])
    model_n = int(model_cfg["num_frames"])
    if data_n != model_n:
        raise ValueError(
            f"data.num_frames ({data_n}) must equal model.num_frames ({model_n})."
        )

    valid_modes = {"mean_only", "frames_only", "frames_anchor", "full", "full_frf_lucky"}
    mode = model_cfg["input_mode"]
    if mode not in valid_modes:
        raise ValueError(f"Invalid model.input_mode='{mode}'. Valid: {sorted(valid_modes)}")

    valid_middle_types = {"naf", "ss2d"}
    middle_type = str(model_cfg.get("middle_type", "naf")).strip().lower()
    if middle_type not in valid_middle_types:
        raise ValueError(
            f"Invalid model.middle_type='{middle_type}'. Valid: {sorted(valid_middle_types)}"
        )

    valid_anchors = {"mean", "frf", "lucky"}
    anchor = model_cfg["anchor_type"]
    if anchor not in valid_anchors:
        raise ValueError(
            f"Invalid model.anchor_type='{anchor}'. Valid: {sorted(valid_anchors)}"
        )

    tile_size = int(eval_cfg["tile_size"])
    overlap = int(eval_cfg["tile_overlap"])
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError(
            f"Invalid tiling config: tile_size={tile_size}, tile_overlap={overlap}"
        )

    total_steps = int(optim_cfg["total_steps"])
    warmup_steps = int(optim_cfg["warmup_steps"])
    if warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError(
            f"warmup_steps ({warmup_steps}) must be in [0, total_steps - 1] ({total_steps})."
        )

    return cfg


def load_config(
    config_path: str | Path,
    variant: Optional[str] = None,
) -> Dict[str, Any]:
    """Loads YAML config, optionally applies ablation preset, and validates constraints."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    chosen_variant = variant if variant is not None else raw.get("experiment", {}).get("variant")
    merged = apply_variant_preset(raw, chosen_variant)
    return validate_config(merged)
