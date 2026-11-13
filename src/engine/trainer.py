"""Single-responsibility training orchestrator with AMP, validation, and W&B logging."""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..contracts import BurstBatch
from ..losses.composite import CompositeLoss
from ..metrics.image_metrics import compute_psnr, compute_ssim
from ..models.resunet import MultiFrameResUNet
from .checkpoint import AtomicCheckpointer
from .wandb_logger import WandbLogger


def resolve_amp_config(device: torch.device, requested: str) -> Tuple[bool, torch.dtype, bool]:
    """
    Resolve `(amp_enabled, autocast_dtype, use_grad_scaler)` for the active device.
    """
    req = requested.lower()
    if device.type != "cuda":
        return False, torch.float32, False

    if req == "bfloat16":
        if torch.cuda.is_bf16_supported():
            return True, torch.bfloat16, False
        warnings.warn(
            "CUDA device does not support bfloat16; falling back to float16 + GradScaler "
            "(LayerNorm2d runs in float32).",
            RuntimeWarning,
            stacklevel=2,
        )
        return True, torch.float16, True
    if req == "float16":
        return True, torch.float16, True
    if req == "float32":
        return False, torch.float32, False
    raise ValueError(f"Unsupported amp_dtype: {requested!r}")


class StaticBurstTrainer:
    """Orchestrates Multi-Frame ResUNet optimization, validation, and atomic checkpointing."""

    def __init__(
        self,
        model: MultiFrameResUNet,
        loss_fn: CompositeLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        checkpointer: AtomicCheckpointer,
        wandb_logger: WandbLogger,
        cfg: Dict[str, Any],
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpointer = checkpointer
        self.wandb_logger = wandb_logger
        self.cfg = cfg
        self.device = device

        optim_cfg = cfg["optim"]
        eval_cfg = cfg["eval"]
        model_cfg = cfg["model"]
        data_cfg = cfg["data"]

        self.total_steps = int(optim_cfg["total_steps"])
        self.grad_accum_steps = max(1, int(data_cfg.get("grad_accum_steps", 1)))
        self.grad_clip_norm = float(optim_cfg.get("grad_clip_norm", 1.0))
        self.stabilize_in_train = bool(model_cfg.get("stabilize_in_train", False))
        self.stabilize_in_eval = bool(model_cfg.get("stabilize_in_eval", False))

        self.log_interval = max(1, int(eval_cfg.get("log_interval", 50)))
        self.val_interval = max(1, int(eval_cfg.get("val_interval", 2000)))
        self.save_interval = max(1, int(eval_cfg.get("save_interval", 2000)))

        amp_enabled, autocast_dtype, use_scaler = resolve_amp_config(
            device, str(optim_cfg.get("amp_dtype", "bfloat16"))
        )
        self.amp_enabled = amp_enabled
        self.autocast_dtype = autocast_dtype
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=True) if use_scaler else None
        )

        self.step: int = 0
        self.best_psnr: float = float("-inf")

    def restore_from_payload(self, payload: Dict[str, Any]) -> None:
        """Update trainer counters after checkpointer loads model/optimizer/scheduler."""
        self.step = int(payload.get("step", 0))
        self.best_psnr = float(payload.get("best_psnr", float("-inf")))
        if self.scaler is not None and payload.get("scaler") is not None:
            self.scaler.load_state_dict(payload["scaler"])

    def _move_batch(self, batch: BurstBatch) -> BurstBatch:
        non_blocking = self.device.type == "cuda"
        return BurstBatch(
            degraded=batch.degraded.to(self.device, non_blocking=non_blocking),
            gt=batch.gt.to(self.device, non_blocking=non_blocking),
            sample_ids=batch.sample_ids,
        )

    @torch.no_grad()
    def validate(self, step: int) -> Dict[str, float]:
        """Run fast centered-crop validation during training and log best model."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        total_anchor_psnr = 0.0
        num_batches = 0
        preview_tensors: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

        for raw_batch in self.val_loader:
            batch = self._move_batch(raw_batch)
            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.autocast_dtype,
                enabled=self.amp_enabled,
            ):
                pred = self.model(batch.degraded, apply_stabilizer=self.stabilize_in_eval)
                loss_out = self.loss_fn(pred.restored, batch.gt)

            total_loss += float(loss_out.total.item())
            total_psnr += compute_psnr(pred.restored, batch.gt)
            total_ssim += compute_ssim(pred.restored, batch.gt)
            total_anchor_psnr += compute_psnr(pred.anchor, batch.gt)
            if preview_tensors is None:
                preview_tensors = (
                    pred.anchor.detach().cpu(),
                    pred.restored.detach().cpu(),
                    batch.gt.detach().cpu(),
                )
            num_batches += 1

        self.model.train()
        if num_batches == 0:
            return {}

        metrics = {
            "val/loss": total_loss / num_batches,
            "val/psnr": total_psnr / num_batches,
            "val/ssim": total_ssim / num_batches,
            "val/anchor_psnr": total_anchor_psnr / num_batches,
            "val/psnr_gain": (total_psnr - total_anchor_psnr) / num_batches,
        }
        self.wandb_logger.log_scalars(metrics, step=step)

        if preview_tensors is not None:
            anchor_cpu, restored_cpu, gt_cpu = preview_tensors
            self.wandb_logger.log_image_grid(
                tag="val/preview",
                anchor=anchor_cpu,
                restored=restored_cpu,
                gt=gt_cpu,
                step=step,
            )

        val_psnr = metrics["val/psnr"]
        if val_psnr > self.best_psnr:
            self.best_psnr = val_psnr
            best_path = self.checkpointer.save(
                filename="best_psnr.pt",
                step=step,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                best_psnr=self.best_psnr,
                config=self.cfg,
                scaler=self.scaler,
                wandb_run_id=self.wandb_logger.run_id,
            )
            self.wandb_logger.log_checkpoint_artifact(
                best_path,
                aliases=["best", f"step-{step}"],
                metadata={"step": step, "val_psnr": val_psnr},
            )

        return metrics

    def _save_last_checkpoint(self, step: int) -> None:
        last_path = self.checkpointer.save(
            filename="last.pt",
            step=step,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            best_psnr=self.best_psnr,
            config=self.cfg,
            scaler=self.scaler,
            wandb_run_id=self.wandb_logger.run_id,
        )
        self.wandb_logger.log_checkpoint_artifact(
            last_path,
            aliases=["latest", f"step-{step}"],
            metadata={"step": step, "best_psnr": self.best_psnr},
        )

    def train(self) -> Dict[str, Any]:
        """Execute training loop up to `total_steps` optimizer updates."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        data_iter = iter(self.train_loader)
        micro_step = 0
        accum_components: Dict[str, float] = {}
        session_start_step = self.step
        start_time = time.perf_counter()

        while self.step < self.total_steps:
            try:
                raw_batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                raw_batch = next(data_iter)

            batch = self._move_batch(raw_batch)
            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.autocast_dtype,
                enabled=self.amp_enabled,
            ):
                # Never clamp model predictions inside training forward/loss
                pred = self.model(batch.degraded, apply_stabilizer=self.stabilize_in_train)
                loss_out = self.loss_fn(pred.restored, batch.gt)
                scaled_loss = loss_out.total / float(self.grad_accum_steps)

            if self.scaler is not None:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            for k, v in loss_out.components.items():
                accum_components[k] = accum_components.get(k, 0.0) + v / float(self.grad_accum_steps)
            micro_step += 1

            if micro_step % self.grad_accum_steps != 0:
                continue

            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)

            grad_norm = float(
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm).item()
            )

            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*lr_scheduler.step.*optimizer.step.*",
                    category=UserWarning,
                )
                self.scheduler.step()
            self.step += 1

            should_log = (self.step % self.log_interval == 0) or (self.step == 1) or (self.step == self.total_steps)
            if should_log:
                elapsed = max(1.0e-6, time.perf_counter() - start_time)
                train_psnr = compute_psnr(pred.restored, batch.gt)
                anchor_psnr = compute_psnr(pred.anchor, batch.gt)
                scalar_logs = {f"train/{k}": v for k, v in accum_components.items()}
                scalar_logs.update(
                    {
                        "train/psnr": train_psnr,
                        "train/anchor_psnr": anchor_psnr,
                        "train/residual_rms": float(pred.residual.detach().float().square().mean().sqrt().item()),
                        "train/lr": float(self.optimizer.param_groups[0]["lr"]),
                        "train/grad_norm": grad_norm,
                        "train/steps_per_sec": float(max(1, self.step - session_start_step)) / elapsed,
                    }
                )
                self.wandb_logger.log_scalars(scalar_logs, step=self.step)
                print(
                    f"[step {self.step}/{self.total_steps}] "
                    f"loss={accum_components.get('total', 0.0):.5f} "
                    f"psnr={train_psnr:.2f}dB anchor={anchor_psnr:.2f}dB "
                    f"lr={scalar_logs['train/lr']:.2e}",
                    flush=True,
                )

            accum_components.clear()

            if (self.step % self.val_interval == 0) or (self.step == self.total_steps):
                val_metrics = self.validate(self.step)
                if val_metrics:
                    print(
                        f"[val step {self.step}] "
                        f"psnr={val_metrics['val/psnr']:.2f}dB "
                        f"(anchor={val_metrics['val/anchor_psnr']:.2f}dB, "
                        f"+{val_metrics['val/psnr_gain']:.2f}dB) "
                        f"ssim={val_metrics['val/ssim']:.4f}",
                        flush=True,
                    )

            if (self.step % self.save_interval == 0) or (self.step == self.total_steps):
                self._save_last_checkpoint(self.step)

        return {
            "final_step": self.step,
            "best_psnr": self.best_psnr,
        }
