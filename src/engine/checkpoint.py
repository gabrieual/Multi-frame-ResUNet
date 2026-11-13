"""Atomic POSIX checkpoint manager implementing ICheckpointer."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class AtomicCheckpointer:
    """
    Saves and loads training checkpoints safely using write-to-temp + `os.replace`
    so preempted Colab or cluster jobs never corrupt `last.pt` or `best_psnr.pt`.
    """

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def resolve(self, filename: str) -> Path:
        return self.checkpoint_dir / filename

    def exists(self, filename: str) -> bool:
        return self.resolve(filename).is_file()

    def save(
        self,
        filename: str,
        step: int,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        best_psnr: float,
        config: Dict[str, Any],
        scaler: Optional[Any] = None,
        wandb_run_id: Optional[str] = None,
    ) -> Path:
        target = self.resolve(filename)
        tmp_path = target.with_name(f".{target.name}.{os.getpid()}.tmp")

        payload: Dict[str, Any] = {
            "step": int(step),
            "best_psnr": float(best_psnr),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "config": config,
            "wandb_run_id": wandb_run_id,
        }

        try:
            with open(tmp_path, "wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, target)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

        return target

    def load(
        self,
        filename: str | Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: str = "cpu",
        scaler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        ckpt_path = Path(filename)
        if not ckpt_path.is_absolute() and not ckpt_path.exists():
            ckpt_path = self.resolve(str(filename))

        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        try:
            payload: Dict[str, Any] = torch.load(
                ckpt_path,
                map_location=map_location,
                weights_only=False,
            )
        except TypeError:
            payload = torch.load(ckpt_path, map_location=map_location)

        model.load_state_dict(payload["model"], strict=True)
        if optimizer is not None and payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload.get("scheduler") is not None:
            saved_lambdas = list(getattr(scheduler, "lr_lambdas", []))
            scheduler.load_state_dict(payload["scheduler"])
            # Preserve current lr_lambdas (e.g. if --total-steps was extended on resume)
            if saved_lambdas and hasattr(scheduler, "lr_lambdas"):
                scheduler.lr_lambdas = saved_lambdas
                if optimizer is not None and hasattr(scheduler, "base_lrs"):
                    for group, base_lr, fn in zip(
                        optimizer.param_groups, scheduler.base_lrs, scheduler.lr_lambdas
                    ):
                        group["lr"] = base_lr * float(fn(scheduler.last_epoch))
                    if hasattr(scheduler, "_last_lr"):
                        scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
        if scaler is not None and payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])

        return payload
