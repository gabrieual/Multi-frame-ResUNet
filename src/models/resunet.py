"""Multi-Frame ResUNet architecture and SOLID dependency-injected builder."""

from typing import Any, Dict, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.contracts import IAnchorBuilder, IStabilizer, ModelPrediction
from src.models.blocks import NAFBlock
from src.models.input_assembler import InputAssembler
from src.registry import ANCHOR_REGISTRY, STABILIZER_REGISTRY
import src.preprocessing  # noqa: F401 - populate stabilizer and anchor registries


class MultiFrameResUNet(nn.Module):
    """
    Static Sequence-to-Image Multi-Frame Residual U-Net using NAFBlocks.
    Always predicts residual relative to `anchor` (J_hat = anchor + residual)
    across all ablation variants (A1..A3c) without clamping inside forward().
    """

    def __init__(
        self,
        stabilizer: IStabilizer,
        anchor_builder: IAnchorBuilder,
        input_assembler: InputAssembler,
        base_ch: int = 48,
        enc_blocks: Sequence[int] = (2, 2, 3),
        middle_blocks: int = 4,
        dec_blocks: Sequence[int] = (2, 2, 2),
        dw_expand: int = 2,
        ffn_expand: int = 2,
        drop_out_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if len(enc_blocks) != len(dec_blocks):
            raise ValueError(
                f"len(enc_blocks)={len(enc_blocks)} must equal len(dec_blocks)={len(dec_blocks)}"
            )

        self.stabilizer = stabilizer
        self.anchor_builder = anchor_builder
        self.input_assembler = input_assembler
        self.num_levels = len(enc_blocks)
        self.pad_divisor = 2 ** self.num_levels

        in_ch = input_assembler.in_channels
        self.stem = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)

        # Encoder hierarchy
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = base_ch
        for num_blk in enc_blocks:
            self.encoders.append(
                nn.Sequential(
                    *[
                        NAFBlock(ch, dw_expand=dw_expand, ffn_expand=ffn_expand, drop_out_rate=drop_out_rate)
                        for _ in range(num_blk)
                    ]
                )
            )
            self.downs.append(nn.Conv2d(ch, ch * 2, kernel_size=2, stride=2))
            ch *= 2

        # Bottleneck
        self.middle = nn.Sequential(
            *[
                NAFBlock(ch, dw_expand=dw_expand, ffn_expand=ffn_expand, drop_out_rate=drop_out_rate)
                for _ in range(middle_blocks)
            ]
        )

        # Decoder hierarchy
        self.ups = nn.ModuleList()
        self.reduces = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for num_blk in dec_blocks:
            out_ch = ch // 2
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(ch, out_ch * 4, kernel_size=1),
                    nn.PixelShuffle(2),
                )
            )
            self.reduces.append(nn.Conv2d(out_ch * 2, out_ch, kernel_size=1))
            self.decoders.append(
                nn.Sequential(
                    *[
                        NAFBlock(out_ch, dw_expand=dw_expand, ffn_expand=ffn_expand, drop_out_rate=drop_out_rate)
                        for _ in range(num_blk)
                    ]
                )
            )
            ch = out_ch

        self.head = nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)
        # Zero initialization guarantees step-0 output starts identically at anchor
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _pad_to_divisor(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        _, _, h, w = x.shape
        d = self.pad_divisor
        pad_h = (d - h % d) % d
        pad_w = (d - w % d) % d
        if pad_h == 0 and pad_w == 0:
            return x, h, w
        pad_mode = "reflect" if (h > pad_h and w > pad_w) else "replicate"
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)
        return x_pad, h, w

    def forward(self, frames: torch.Tensor, apply_stabilizer: bool = False) -> ModelPrediction:
        """
        Args:
            frames: [B, N, 3, H, W] in [0, 1]
            apply_stabilizer: Whether to run injected global stabilizer (false in train, optional in eval)
        """
        with torch.no_grad():
            stab_frames = self.stabilizer.stabilize(frames) if apply_stabilizer else frames
            anchor = self.anchor_builder.build(stab_frames)
            x_in, variance_map = self.input_assembler.assemble(stab_frames, anchor)

        x_pad, orig_h, orig_w = self._pad_to_divisor(x_in)

        feat = self.stem(x_pad)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            feat = enc(feat)
            skips.append(feat)
            feat = down(feat)

        feat = self.middle(feat)

        for up, reduce, dec, skip in zip(
            self.ups, self.reduces, self.decoders, reversed(skips)
        ):
            feat = up(feat)
            feat = reduce(torch.cat([feat, skip], dim=1))
            feat = dec(feat)

        residual = self.head(feat)[:, :, :orig_h, :orig_w]
        restored = anchor + residual  # Un-clamped to preserve gradient flow at [0, 1] boundaries

        return ModelPrediction(
            restored=restored,
            anchor=anchor,
            residual=residual,
            variance_map=variance_map,
            stabilized_frames=stab_frames,
        )


def build_model(cfg: Dict[str, Any]) -> MultiFrameResUNet:
    """Factory wiring dependencies via Registries (Dependency Inversion)."""
    model_cfg = cfg["model"]

    stabilizer_name = "fft_global" if model_cfg.get("stabilize_in_eval", False) else "none"
    stabilizer = STABILIZER_REGISTRY.build(stabilizer_name)

    anchor_type = model_cfg["anchor_type"]
    if anchor_type == "frf":
        anchor_builder = ANCHOR_REGISTRY.build(
            "frf",
            bandwidth_h=float(model_cfg.get("frf_bandwidth_h", 0.05)),
            sigma=float(model_cfg.get("frf_sigma", 0.5)),
        )
    else:
        anchor_builder = ANCHOR_REGISTRY.build("mean")

    assembler = InputAssembler(
        num_frames=int(model_cfg["num_frames"]),
        input_mode=str(model_cfg["input_mode"]),
    )

    return MultiFrameResUNet(
        stabilizer=stabilizer,
        anchor_builder=anchor_builder,
        input_assembler=assembler,
        base_ch=int(model_cfg.get("base_ch", 48)),
        enc_blocks=list(model_cfg.get("enc_blocks", [2, 2, 3])),
        middle_blocks=int(model_cfg.get("middle_blocks", 4)),
        dec_blocks=list(model_cfg.get("dec_blocks", [2, 2, 2])),
        dw_expand=int(model_cfg.get("dw_expand", 2)),
        ffn_expand=int(model_cfg.get("ffn_expand", 2)),
        drop_out_rate=float(model_cfg.get("drop_out_rate", 0.0)),
    )
