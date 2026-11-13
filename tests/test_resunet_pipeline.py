"""Comprehensive unit and end-to-end tests for the standalone ResUNet package."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np
from PIL import Image
import torch

RESUNET_ROOT = Path(__file__).resolve().parents[1]
if str(RESUNET_ROOT) not in sys.path:
    sys.path.insert(0, str(RESUNET_ROOT))

from scripts.prepare_npy_cache import _convert_sample  # noqa: E402
from src.config import ABLATION_PRESETS, load_config, validate_config  # noqa: E402
from src.data import StaticBurstDataset, burst_collate_fn  # noqa: E402
from src.engine import (  # noqa: E402
    AtomicCheckpointer,
    StaticBurstTrainer,
    TiledInference,
    WandbLogger,
    WarmupCosineLambda,
    build_optimizer_and_scheduler,
)
from src.engine.optimizer_factory import split_decay_parameters  # noqa: E402
from src.losses import build_loss  # noqa: E402
from src.metrics import compute_psnr, compute_ssim  # noqa: E402
from src.models import LayerNorm2d, build_model  # noqa: E402
import src.preprocessing  # noqa: E402, F401
from src.registry import ANCHOR_REGISTRY, STABILIZER_REGISTRY  # noqa: E402


def _make_mini_cfg(
    train_root: str = "/tmp/train",
    val_root: str = "/tmp/val",
    output_dir: str = "/tmp/out",
    variant: str = "A3c",
) -> dict:
    cfg = load_config(RESUNET_ROOT / "configs" / "default.yaml", variant=variant)
    cfg["experiment"]["output_dir"] = output_dir
    cfg["experiment"]["use_wandb"] = False
    cfg["data"]["train_root"] = train_root
    cfg["data"]["val_root"] = val_root
    cfg["data"]["num_frames"] = 4
    cfg["data"]["crop_size"] = 64
    cfg["data"]["batch_size"] = 2
    cfg["data"]["num_workers"] = 0
    cfg["model"]["num_frames"] = 4
    cfg["model"]["base_ch"] = 16
    cfg["model"]["enc_blocks"] = [1, 1, 1]
    cfg["model"]["middle_blocks"] = 1
    cfg["model"]["dec_blocks"] = [1, 1, 1]
    cfg["optim"]["total_steps"] = 4
    cfg["optim"]["warmup_steps"] = 1
    cfg["optim"]["amp_dtype"] = "float32"
    cfg["eval"]["tile_size"] = 64
    cfg["eval"]["tile_overlap"] = 16
    cfg["eval"]["val_interval"] = 2
    cfg["eval"]["save_interval"] = 2
    cfg["eval"]["log_interval"] = 1
    return validate_config(cfg)


def _write_png_dataset(root: Path, num_samples: int = 2, num_frames: int = 6, h: int = 80, w: int = 96) -> None:
    rng = np.random.default_rng(123)
    for idx in range(num_samples):
        sid = f"seq_{idx:02d}"
        turb_dir = root / "turb" / sid
        turb_dir.mkdir(parents=True, exist_ok=True)
        gt_dir = root / "gt"
        gt_dir.mkdir(parents=True, exist_ok=True)

        base = rng.integers(30, 220, size=(h, w, 3), dtype=np.uint8)
        Image.fromarray(base).save(gt_dir / f"{sid}.png")
        for f_idx in range(num_frames):
            noise = rng.integers(-12, 13, size=(h, w, 3), dtype=np.int16)
            frame = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            Image.fromarray(frame).save(turb_dir / f"frame_{f_idx:03d}.png")


class TestResUNetPipeline(unittest.TestCase):
    def test_variants_input_channels_and_zero_init_anchor(self) -> None:
        expected_channels = {
            "A1": 3,
            "A2": 12,        # 3 * 4 frames
            "A3a": 15,       # 3 * 4 + 3
            "A3b": 16,       # 3 * 4 + 4
            "A3b-FRF": 16,   # 3 * 4 + 4
            "A3c": 16,       # 3 * 4 + 4
        }
        burst = torch.rand(2, 4, 3, 64, 64)
        for variant in ABLATION_PRESETS:
            cfg = _make_mini_cfg(variant=variant)
            model = build_model(cfg)
            self.assertEqual(model.input_assembler.in_channels, expected_channels[variant])
            pred = model(burst)
            self.assertEqual(pred.restored.shape, (2, 3, 64, 64))
            self.assertEqual(pred.anchor.shape, (2, 3, 64, 64))
            self.assertEqual(pred.residual.shape, (2, 3, 64, 64))
            # Zero-initialized output head implies residual == 0 and restored == anchor initially
            self.assertLess(pred.residual.abs().max().item(), 1.0e-7)
            self.assertTrue(torch.allclose(pred.restored, pred.anchor, atol=1.0e-7))

            # Verify gradient flow
            loss = pred.restored.square().mean()
            loss.backward()
            has_nonzero_grad = any(
                p.grad is not None and p.grad.abs().sum().item() > 0.0
                for p in model.parameters()
            )
            self.assertTrue(has_nonzero_grad)

    def test_spec_parameter_counts(self) -> None:
        cfg_48 = load_config(RESUNET_ROOT / "configs" / "default.yaml", variant="A3c")
        cfg_48["model"]["base_ch"] = 48
        p48 = sum(p.numel() for p in build_model(cfg_48).parameters())

        cfg_64 = copy.deepcopy(cfg_48)
        cfg_64["model"]["base_ch"] = 64
        p64 = sum(p.numel() for p in build_model(cfg_64).parameters())

        print(f"\n[verified param count] base_ch=48 -> {p48:,} | base_ch=64 -> {p64:,}")
        self.assertGreater(p48, 5_500_000)
        self.assertLess(p48, 7_500_000)
        self.assertGreater(p64, 10_500_000)
        self.assertLess(p64, 13_000_000)

    def test_layernorm2d_fp16_safety(self) -> None:
        norm = LayerNorm2d(channels=16)
        x = (torch.randn(2, 16, 16, 16) * 5000.0).to(torch.float16)
        out = norm(x)
        self.assertEqual(out.dtype, torch.float16)
        self.assertTrue(torch.isfinite(out).all().item())

    def test_fft_stabilizer_and_frf_anchor(self) -> None:
        # Synthetic bright gaussian reference blob
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, 64),
            torch.linspace(-1.0, 1.0, 64),
            indexing="ij",
        )
        ref = torch.exp(-((yy + 0.1) ** 2 + (xx - 0.1) ** 2) / 0.04).view(1, 1, 64, 64).repeat(1, 3, 1, 1)
        shifted = torch.roll(ref, shifts=(5, -7), dims=(-2, -1))
        burst = torch.stack([ref.squeeze(0), shifted.squeeze(0)], dim=0).unsqueeze(0)  # [1, 2, 3, 64, 64]

        stabilizer = STABILIZER_REGISTRY.build("fft_global")
        aligned = stabilizer.stabilize(burst)
        before_err = (burst[:, 1] - burst[:, 0]).abs().mean().item()
        after_err = (aligned[:, 1] - aligned[:, 0]).abs().mean().item()
        self.assertLess(after_err, before_err * 0.1)

        # FRF Parzen mode anchor resists transient outlier displacement better than temporal mean
        clean = (torch.sin(xx * 18.0) * torch.cos(yy * 18.0) * 0.35 + 0.5).view(1, 1, 64, 64).repeat(1, 3, 1, 1)
        outlier = 1.0 - clean
        mixed = torch.cat([clean, clean, clean, outlier], dim=0).unsqueeze(0)  # [1, 4, 3, 64, 64]
        mean_out = ANCHOR_REGISTRY.build("mean").build(mixed)
        frf_out = ANCHOR_REGISTRY.build("frf", bandwidth_h=0.05, sigma=0.5).build(mixed)
        self.assertGreater(compute_psnr(frf_out, clean), compute_psnr(mean_out, clean))

    def test_optimizer_weight_decay_and_lr_schedule(self) -> None:
        cfg = _make_mini_cfg()
        cfg["optim"]["lr"] = 2.0e-4
        cfg["optim"]["min_lr"] = 1.0e-6
        cfg["optim"]["total_steps"] = 100
        cfg["optim"]["warmup_steps"] = 10
        model = build_model(cfg)
        optimizer, _ = build_optimizer_and_scheduler(model, cfg)

        decay_params, no_decay_params = split_decay_parameters(model)
        no_decay_ids = {id(p) for p in no_decay_params}
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 1.0e-4)
        self.assertEqual(optimizer.param_groups[1]["weight_decay"], 0.0)

        for name, param in model.named_parameters():
            leaf = name.rsplit(".", 1)[-1]
            if leaf in {"beta", "gamma", "bias"} or "norm" in name:
                self.assertIn(id(param), no_decay_ids, f"{name} unexpectedly decayed")
        self.assertGreater(len(decay_params), 0)

        sched_fn = WarmupCosineLambda(base_lr=2.0e-4, min_lr=1.0e-6, total_steps=100, warmup_steps=10)
        self.assertAlmostEqual(sched_fn(0) * 2.0e-4, 2.0e-5, places=10)
        self.assertAlmostEqual(sched_fn(9) * 2.0e-4, 2.0e-4, places=10)
        self.assertAlmostEqual(sched_fn(10) * 2.0e-4, 2.0e-4, places=10)
        self.assertAlmostEqual(sched_fn(55) * 2.0e-4, 0.5 * (2.0e-4 + 1.0e-6), places=10)
        self.assertAlmostEqual(sched_fn(100) * 2.0e-4, 1.0e-6, places=10)

    def test_config_strict_validation(self) -> None:
        cfg = _make_mini_cfg()
        cfg["data"]["allow_synthetic_fallback"] = True
        with self.assertRaises(ValueError):
            validate_config(cfg)

        cfg2 = _make_mini_cfg()
        cfg2["model"]["num_frames"] = 99
        with self.assertRaises(ValueError):
            validate_config(cfg2)

        with self.assertRaises(ValueError):
            StaticBurstDataset("/tmp/does_not_exist_12345", allow_synthetic_fallback=True)
        with self.assertRaises(FileNotFoundError):
            StaticBurstDataset("/tmp/does_not_exist_12345", allow_synthetic_fallback=False)

    def test_dataset_layouts_sampling_and_npy_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_root = tmp_path / "raw"
            npy_root = tmp_path / "npy"
            _write_png_dataset(raw_root, num_samples=2, num_frames=10, h=80, w=96)

            records = StaticBurstDataset._discover_samples(raw_root)
            self.assertEqual(len(records), 2)
            for rec in records:
                _convert_sample(rec, npy_root, overwrite=True)

            train_ds = StaticBurstDataset(
                npy_root, num_frames=4, crop_size=64, is_train=True, return_all_frames=False
            )
            frames, gt, sid = train_ds[0]
            self.assertEqual(frames.shape, (4, 3, 64, 64))
            self.assertEqual(gt.shape, (3, 64, 64))
            self.assertTrue(sid.startswith("seq_"))

            # Verify training random sampling does not sort indices
            saw_unsorted = False
            for _ in range(25):
                idx = train_ds._select_indices(10)
                if idx != sorted(idx):
                    saw_unsorted = True
                    break
            self.assertTrue(saw_unsorted)

            # Verify eval full-sequence loader without cropping
            eval_ds = StaticBurstDataset(
                npy_root, num_frames=4, crop_size=0, is_train=False, return_all_frames=True
            )
            full_seq, full_gt, _ = eval_ds[0]
            self.assertEqual(full_seq.shape, (10, 3, 80, 96))
            self.assertEqual(full_gt.shape, (3, 80, 96))

    def test_tiled_inference_various_resolutions_and_global_stabilization(self) -> None:
        cfg = _make_mini_cfg()
        model = build_model(cfg)
        model.eval()
        evaluator = TiledInference(tile_size=64, overlap=16, max_eval_windows=3, stabilize_in_eval=True)

        # Spy on stabilizer calls to ensure full-frame stabilize runs once per burst
        orig_stabilize = model.stabilizer.stabilize
        call_shapes = []

        def tracked_stabilize(x: torch.Tensor) -> torch.Tensor:
            call_shapes.append(tuple(x.shape))
            return orig_stabilize(x)

        model.stabilizer.stabilize = tracked_stabilize  # type: ignore

        small = torch.rand(1, 4, 3, 40, 50)
        pred_small = evaluator.predict_burst(model, small)
        self.assertEqual(pred_small.restored.shape, (1, 3, 40, 50))
        self.assertEqual(call_shapes[-1], (1, 4, 3, 40, 50))

        large = torch.rand(1, 4, 3, 135, 173)
        pred_large = evaluator.predict_burst(model, large)
        self.assertEqual(pred_large.restored.shape, (1, 3, 135, 173))
        self.assertEqual(call_shapes[-1], (1, 4, 3, 135, 173))
        self.assertEqual(len(call_shapes), 2)
        self.assertTrue(torch.isfinite(pred_large.restored).all().item())

        seq = torch.rand(9, 3, 100, 110)
        pred_seq = evaluator.predict_sequence(model, seq, num_frames=4)
        self.assertEqual(pred_seq.restored.shape, (1, 3, 100, 110))

    def test_loss_metrics_and_wandb_logger_mock(self) -> None:
        cfg = _make_mini_cfg(variant="A3c")
        loss_fn = build_loss(cfg)
        pred = torch.rand(2, 3, 64, 64, requires_grad=True)
        target = torch.rand(2, 3, 64, 64)
        breakdown = loss_fn(pred, target)
        self.assertEqual(set(breakdown.components.keys()), {"charbonnier", "fft", "edge", "total"})
        breakdown.total.backward()
        self.assertIsNotNone(pred.grad)

        clean = torch.rand(1, 3, 32, 32)
        noisy = (clean + 0.05).clamp(0.0, 1.0)
        self.assertGreater(compute_psnr(clean, clean), compute_psnr(noisy, clean))
        self.assertGreater(compute_ssim(clean, clean), compute_ssim(noisy, clean))

        # Mocked W&B integration test
        fake_run = MagicMock()
        fake_run.id = "wandb-test-id"
        fake_wandb = MagicMock()
        fake_wandb.init.return_value = fake_run
        fake_wandb.Image.side_effect = lambda arr, caption="": ("img", arr.shape, caption)
        fake_wandb.Artifact.return_value = MagicMock()

        cfg_wb = copy.deepcopy(cfg)
        cfg_wb["experiment"]["use_wandb"] = True
        wb = WandbLogger(cfg_wb, _wandb_module=fake_wandb)
        wb.start_run(cfg_wb)
        wb.log_scalars({"train/loss": 0.1}, step=1)
        wb.log_image_grid("val/preview", clean, noisy, clean, step=1)
        fake_run.log.assert_called()
        wb.finish()
        fake_run.finish.assert_called_once()

    def test_end_to_end_train_resume_and_eval_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_train = tmp_path / "raw_train"
            raw_val = tmp_path / "raw_val"
            npy_train = tmp_path / "npy_train"
            npy_val = tmp_path / "npy_val"
            out_dir = tmp_path / "runs"

            _write_png_dataset(raw_train, num_samples=2, num_frames=6, h=80, w=96)
            _write_png_dataset(raw_val, num_samples=1, num_frames=6, h=80, w=96)

            for rec in StaticBurstDataset._discover_samples(raw_train):
                _convert_sample(rec, npy_train)
            for rec in StaticBurstDataset._discover_samples(raw_val):
                _convert_sample(rec, npy_val)

            cfg = _make_mini_cfg(str(npy_train), str(npy_val), str(out_dir), variant="A3c")
            cfg["optim"]["total_steps"] = 2
            cfg["eval"]["val_interval"] = 2
            cfg["eval"]["save_interval"] = 2
            cfg_path = tmp_path / "mini.yaml"
            import yaml
            with open(cfg_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(cfg, handle)

            # 1) CLI train 2 steps
            subprocess.run(
                [
                    sys.executable,
                    str(RESUNET_ROOT / "scripts" / "train.py"),
                    "--config",
                    str(cfg_path),
                    "--wandb-project",
                    "unit-test-proj",
                    "--wandb-entity",
                    "unit-test-entity",
                    "--num-frames",
                    "4",
                    "--crop-size",
                    "64",
                    "--no-wandb",
                ],
                cwd=str(RESUNET_ROOT),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            ckpt_path = out_dir / cfg["experiment"]["name"] / "checkpoints" / "last.pt"
            self.assertTrue(ckpt_path.is_file())

            # 2) Auto-resume to step 4
            subprocess.run(
                [
                    sys.executable,
                    str(RESUNET_ROOT / "scripts" / "train.py"),
                    "--config",
                    str(cfg_path),
                    "--total-steps",
                    "4",
                    "--auto-resume",
                    "--no-wandb",
                ],
                cwd=str(RESUNET_ROOT),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            resumed_payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.assertEqual(resumed_payload["step"], 4)

            # 3) CLI eval A0-mean and A3c with saved PNGs + JSON report
            json_a0 = tmp_path / "eval_a0.json"
            json_a3c = tmp_path / "eval_a3c.json"
            vis_dir = tmp_path / "vis"

            subprocess.run(
                [
                    sys.executable,
                    str(RESUNET_ROOT / "scripts" / "eval.py"),
                    "--config",
                    str(cfg_path),
                    "--variant",
                    "A0-mean",
                    "--test-root",
                    str(npy_val),
                    "--output-json",
                    str(json_a0),
                ],
                cwd=str(RESUNET_ROOT),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(RESUNET_ROOT / "scripts" / "eval.py"),
                    "--config",
                    str(cfg_path),
                    "--variant",
                    "A0-frf",
                    "--stabilize",
                    "--test-root",
                    str(npy_val),
                ],
                cwd=str(RESUNET_ROOT),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(RESUNET_ROOT / "scripts" / "eval.py"),
                    "--checkpoint",
                    str(ckpt_path),
                    "--test-root",
                    str(npy_val),
                    "--save-dir",
                    str(vis_dir),
                    "--output-json",
                    str(json_a3c),
                ],
                cwd=str(RESUNET_ROOT),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            with open(json_a0, encoding="utf-8") as h0:
                report_a0 = json.load(h0)
            with open(json_a3c, encoding="utf-8") as h3:
                report_a3c = json.load(h3)

            self.assertEqual(report_a0["num_samples"], 1)
            self.assertEqual(report_a3c["num_samples"], 1)
            self.assertTrue((vis_dir / "seq_00" / "restored.png").is_file())
            self.assertTrue((vis_dir / "seq_00" / "comparison_anchor_restored_gt.png").is_file())


if __name__ == "__main__":
    unittest.main()
