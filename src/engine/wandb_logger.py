"""Isolated Weights & Biases logger with safe offline fallback and dependency injection."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch


class WandbLogger:
    """
    Single-responsibility wrapper around `wandb` for experiment telemetry,
    side-by-side restoration previews, and model artifact checkpointing.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        _wandb_module: Optional[Any] = None,
    ) -> None:
        exp_cfg = cfg.get("experiment", {})
        self.enabled: bool = bool(exp_cfg.get("use_wandb", True))
        self.project: str = str(exp_cfg.get("wandb_project", "turbulence-mitigation-static"))
        self.entity: Optional[str] = exp_cfg.get("wandb_entity")
        self.run_name: str = str(exp_cfg.get("name", "multiframe_resunet"))
        self.variant: str = str(exp_cfg.get("variant", "custom"))
        self.log_images: bool = bool(exp_cfg.get("wandb_log_images", True))
        self.upload_checkpoints: bool = bool(exp_cfg.get("wandb_upload_checkpoints", True))
        self._artifact_base: str = f"{self.run_name}-ckpt"
        self._wandb: Optional[Any] = _wandb_module
        self.run: Optional[Any] = None

        if self.enabled and self._wandb is None:
            try:
                import wandb  # type: ignore
            except ImportError as exc:
                raise ModuleNotFoundError(
                    "experiment.use_wandb is true, but `wandb` is not installed. "
                    "Install it with `pip install wandb` or run with `--no-wandb`."
                ) from exc
            self._wandb = wandb

    @property
    def run_id(self) -> Optional[str]:
        if self.run is not None:
            return getattr(self.run, "id", None)
        return None

    def start_run(
        self,
        cfg: Dict[str, Any],
        run_id: Optional[str] = None,
        resume: str = "allow",
    ) -> Optional[str]:
        """Initialize the W&B run if logging is enabled."""
        if not self.enabled or self._wandb is None:
            return None

        init_kwargs: Dict[str, Any] = {
            "project": self.project,
            "name": self.run_name,
            "config": cfg,
            "tags": [self.variant, "static-s2i", "resunet"],
            "resume": resume,
        }
        if self.entity:
            init_kwargs["entity"] = self.entity
        if run_id:
            init_kwargs["id"] = run_id

        self.run = self._wandb.init(**init_kwargs)
        return self.run_id

    def log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        """Log a dictionary of numeric metrics at `step`."""
        if not self.enabled or self.run is None or not metrics:
            return
        self.run.log(metrics, step=int(step))

    @torch.no_grad()
    def log_image_grid(
        self,
        tag: str,
        anchor: torch.Tensor,
        restored: torch.Tensor,
        gt: torch.Tensor,
        step: int,
        max_items: int = 2,
        caption_prefix: str = "Anchor | Restored | GT",
    ) -> None:
        """Log side-by-side `[Anchor | Restored | GT]` comparison strips clamped to [0, 1]."""
        if not self.enabled or not self.log_images or self.run is None or self._wandb is None:
            return

        if anchor.ndim == 3:
            anchor = anchor.unsqueeze(0)
            restored = restored.unsqueeze(0)
            gt = gt.unsqueeze(0)

        num_items = min(max_items, anchor.shape[0])
        images = []
        for idx in range(num_items):
            strip = torch.cat(
                [
                    anchor[idx].detach().cpu().float().clamp(0.0, 1.0),
                    restored[idx].detach().cpu().float().clamp(0.0, 1.0),
                    gt[idx].detach().cpu().float().clamp(0.0, 1.0),
                ],
                dim=2,  # Concatenate horizontally along W
            )
            arr = (strip.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
            images.append(self._wandb.Image(arr, caption=f"{caption_prefix} (sample {idx})"))

        self.run.log({tag: images}, step=int(step))

    @staticmethod
    def _sanitize_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Strip or null non-finite floats because W&B validates metadata with allow_nan=False."""
        clean: Dict[str, Any] = {}
        for key, val in (metadata or {}).items():
            if isinstance(val, float) and not math.isfinite(val):
                clean[key] = None
            else:
                clean[key] = val
        return clean

    def log_checkpoint_artifact(
        self,
        checkpoint_path: str | Path,
        aliases: Sequence[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Upload a checkpoint file as a versioned W&B Artifact."""
        if not self.enabled or not self.upload_checkpoints or self.run is None or self._wandb is None:
            return

        path = Path(checkpoint_path)
        if not path.is_file():
            return

        artifact = self._wandb.Artifact(
            name=self._artifact_base,
            type="model",
            metadata=self._sanitize_metadata(metadata),
        )
        artifact.add_file(str(path), name=path.name)
        self.run.log_artifact(artifact, aliases=list(aliases))

    def download_checkpoint_artifact(
        self,
        target_dir: str | Path,
        filename: str = "last.pt",
        alias: str = "latest",
    ) -> Optional[Path]:
        """
        Attempt to pull the latest checkpoint artifact from W&B for `--auto-resume`.
        Returns None cleanly if W&B is disabled or no artifact exists yet.
        """
        if not self.enabled or self._wandb is None:
            return None

        dest_dir = Path(target_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        artifact_ref = f"{self.entity}/{self.project}/{self._artifact_base}:{alias}" if self.entity else f"{self.project}/{self._artifact_base}:{alias}"

        try:
            if self.run is not None:
                artifact = self.run.use_artifact(artifact_ref, type="model")
            else:
                api = self._wandb.Api()
                artifact = api.artifact(artifact_ref, type="model")
            downloaded_dir = Path(artifact.download(root=str(dest_dir)))
        except Exception:
            return None

        candidate = downloaded_dir / filename
        if candidate.is_file():
            return candidate
        pt_files = sorted(downloaded_dir.glob("*.pt"))
        return pt_files[0] if pt_files else None

    def finish(self, exit_code: int = 0) -> None:
        """Close the active W&B run cleanly (passing non-zero exit_code on failure)."""
        if self.enabled and self.run is not None:
            if exit_code == 0:
                self.run.finish()
            else:
                self.run.finish(exit_code=int(exit_code))
            self.run = None
