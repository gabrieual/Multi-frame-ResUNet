"""Training CLI entrypoint for Multi-Frame ResUNet."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

RESUNET_ROOT = Path(__file__).resolve().parents[1]
if str(RESUNET_ROOT) not in sys.path:
    sys.path.insert(0, str(RESUNET_ROOT))

from src.config import ABLATION_PRESETS, load_config, validate_config  # noqa: E402
from src.data import StaticBurstDataset, burst_collate_fn  # noqa: E402
from src.engine import (  # noqa: E402
    AtomicCheckpointer,
    StaticBurstTrainer,
    WandbLogger,
    build_optimizer_and_scheduler,
)
from src.losses import build_loss  # noqa: E402
from src.models import build_model  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    """Ensure each forked DataLoader worker has an independent random/numpy stream."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed + worker_id)
    np.random.seed((worker_seed + worker_id) % (2 ** 32))


def _make_dataloader(
    dataset: StaticBurstDataset,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    data_cfg: Dict[str, Any],
    device: torch.device,
) -> DataLoader:
    num_workers = int(data_cfg.get("num_workers", 4))
    kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "num_workers": num_workers,
        "pin_memory": bool(data_cfg.get("pin_memory", True)) and device.type == "cuda",
        "collate_fn": burst_collate_fn,
        "worker_init_fn": _seed_worker,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
    return DataLoader(dataset, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Multi-Frame ResUNet on static bursts.")
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
        choices=sorted(ABLATION_PRESETS.keys()),
        help="Optional ablation preset override.",
    )
    parser.add_argument("--train-root", type=str, default=None, help="Override data.train_root")
    parser.add_argument("--val-root", type=str, default=None, help="Override data.val_root")
    parser.add_argument("--output-dir", type=str, default=None, help="Override experiment.output_dir")
    parser.add_argument("--name", type=str, default=None, help="Override experiment.name")
    parser.add_argument("--base-ch", type=int, default=None, help="Override model.base_ch (48 or 64)")
    parser.add_argument("--num-frames", type=int, default=None, help="Override data/model num_frames")
    parser.add_argument("--crop-size", type=int, default=None, help="Override data.crop_size")
    parser.add_argument("--total-steps", type=int, default=None, help="Override optim.total_steps")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Override optim.warmup_steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Override data.batch_size")
    parser.add_argument("--num-workers", type=int, default=None, help="Override data.num_workers")
    parser.add_argument("--val-interval", type=int, default=None, help="Override eval.val_interval")
    parser.add_argument("--save-interval", type=int, default=None, help="Override eval.save_interval")
    parser.add_argument("--log-interval", type=int, default=None, help="Override eval.log_interval")
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default=None,
        choices=["bfloat16", "float16", "float32"],
        help="Override optim.amp_dtype",
    )
    parser.add_argument("--ssim-weight", type=float, default=None, help="Override loss.ssim_weight")
    parser.add_argument("--seed", type=int, default=None, help="Override experiment.seed")
    parser.add_argument("--wandb-project", type=str, default=None, help="Override experiment.wandb_project")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Override experiment.wandb_entity")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume from local last.pt or latest W&B checkpoint artifact if found.",
    )
    parser.add_argument("--resume-path", type=Path, default=None, help="Explicit checkpoint to resume")
    return parser.parse_args()


def _apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.train_root is not None:
        cfg["data"]["train_root"] = args.train_root
    if args.val_root is not None:
        cfg["data"]["val_root"] = args.val_root
    if args.output_dir is not None:
        cfg["experiment"]["output_dir"] = args.output_dir
    if args.name is not None:
        cfg["experiment"]["name"] = args.name
    elif args.variant is not None:
        cfg["experiment"]["name"] = f"multiframe_resunet_{args.variant.lower()}"
    if args.base_ch is not None:
        cfg["model"]["base_ch"] = args.base_ch
    if args.num_frames is not None:
        cfg["data"]["num_frames"] = args.num_frames
        cfg["model"]["num_frames"] = args.num_frames
    if args.crop_size is not None:
        cfg["data"]["crop_size"] = args.crop_size
    if args.total_steps is not None:
        cfg["optim"]["total_steps"] = args.total_steps
    if args.warmup_steps is not None:
        cfg["optim"]["warmup_steps"] = args.warmup_steps
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.val_interval is not None:
        cfg["eval"]["val_interval"] = args.val_interval
    if args.save_interval is not None:
        cfg["eval"]["save_interval"] = args.save_interval
    if args.log_interval is not None:
        cfg["eval"]["log_interval"] = args.log_interval
    if args.amp_dtype is not None:
        cfg["optim"]["amp_dtype"] = args.amp_dtype
    if args.ssim_weight is not None:
        cfg["loss"]["ssim_weight"] = args.ssim_weight
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed
    if args.wandb_project is not None:
        cfg["experiment"]["wandb_project"] = args.wandb_project
    if args.wandb_entity is not None:
        cfg["experiment"]["wandb_entity"] = args.wandb_entity
    if args.no_wandb:
        cfg["experiment"]["use_wandb"] = False
    return validate_config(cfg)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, variant=args.variant)
    cfg = _apply_cli_overrides(cfg, args)

    seed_everything(int(cfg["experiment"].get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(cfg["experiment"]["output_dir"]) / cfg["experiment"]["name"]
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config_resolved.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    data_cfg = cfg["data"]
    allow_fallback = bool(data_cfg.get("allow_synthetic_fallback", False))
    train_ds = StaticBurstDataset(
        root_dir=data_cfg["train_root"],
        num_frames=int(data_cfg["num_frames"]),
        crop_size=int(data_cfg["crop_size"]),
        is_train=True,
        allow_synthetic_fallback=allow_fallback,
        return_all_frames=False,
    )
    train_loader = _make_dataloader(
        train_ds,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=True,
        drop_last=len(train_ds) >= int(data_cfg["batch_size"]),
        data_cfg=data_cfg,
        device=device,
    )

    val_loader: Optional[DataLoader] = None
    val_root = data_cfg.get("val_root")
    if val_root:
        val_ds = StaticBurstDataset(
            root_dir=val_root,
            num_frames=int(data_cfg["num_frames"]),
            crop_size=int(data_cfg["crop_size"]),
            is_train=False,
            allow_synthetic_fallback=allow_fallback,
            return_all_frames=False,
        )
        val_loader = _make_dataloader(
            val_ds,
            batch_size=min(int(data_cfg["batch_size"]), len(val_ds)),
            shuffle=False,
            drop_last=False,
            data_cfg=data_cfg,
            device=device,
        )

    model = build_model(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Variant={cfg['experiment'].get('variant')} | Params={total_params:,} | "
        f"Train samples={len(train_ds)} | Device={device}",
        flush=True,
    )

    loss_fn = build_loss(cfg)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)
    checkpointer = AtomicCheckpointer(ckpt_dir)
    wandb_logger = WandbLogger(cfg)

    trainer = StaticBurstTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpointer=checkpointer,
        wandb_logger=wandb_logger,
        cfg=cfg,
        device=device,
    )

    resume_file: Optional[Path] = args.resume_path
    if resume_file is None and args.auto_resume:
        if checkpointer.exists("last.pt"):
            resume_file = checkpointer.resolve("last.pt")
        else:
            resume_file = wandb_logger.download_checkpoint_artifact(ckpt_dir)

    resumed_wandb_id: Optional[str] = None
    if resume_file is not None:
        payload = checkpointer.load(
            resume_file,
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            map_location=str(device),
            scaler=trainer.scaler,
        )
        trainer.restore_from_payload(payload)
        resumed_wandb_id = payload.get("wandb_run_id")
        print(
            f"Resumed from {resume_file} at step={trainer.step}, best_psnr={trainer.best_psnr:.2f}dB",
            flush=True,
        )

    wandb_logger.start_run(cfg, run_id=resumed_wandb_id, resume="allow")
    try:
        summary = trainer.train()
        print(
            f"Training finished at step={summary['final_step']} | best_psnr={summary['best_psnr']:.2f}dB",
            flush=True,
        )
    finally:
        wandb_logger.finish()


if __name__ == "__main__":
    main()
