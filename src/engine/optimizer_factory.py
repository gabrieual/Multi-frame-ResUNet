"""AdamW parameter grouping and warmup + cosine decay scheduler factory."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from ..models.blocks import LayerNorm2d

_NO_DECAY_PARAM_NAMES = {"beta", "gamma", "bias"}


class WarmupCosineLambda:
    """
    Callable LR multiplier implementing linear warmup followed by cosine decay
    strictly over T_max = total_steps - warmup_steps steps down to min_lr.
    """

    def __init__(
        self,
        base_lr: float,
        min_lr: float,
        total_steps: int,
        warmup_steps: int,
    ) -> None:
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if warmup_steps < 0 or warmup_steps >= total_steps:
            raise ValueError(
                f"Require 0 <= warmup_steps < total_steps, got warmup={warmup_steps}, total={total_steps}"
            )
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.min_ratio = self.min_lr / self.base_lr if self.base_lr > 0.0 else 0.0
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.t_max = max(1, self.total_steps - self.warmup_steps)

    def __call__(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            # Linear warmup starting > 0 at step=0 and hitting 1.0 at step=warmup_steps - 1.
            return float(step + 1) / float(self.warmup_steps)

        progress = min(1.0, max(0.0, float(step - self.warmup_steps) / float(self.t_max)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_ratio + (1.0 - self.min_ratio) * cosine


def split_decay_parameters(
    model: nn.Module,
) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """
    Separate trainable parameters into decay and no_decay groups.

    LayerNorm2d parameters (weight and bias are 4D tensors), biases,
    and NAFBlock residual scalars (beta, gamma) are excluded from weight decay.
    """
    norm_param_ids = {
        id(param)
        for module in model.modules()
        if isinstance(module, LayerNorm2d)
        for param in module.parameters(recurse=False)
    }

    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        leaf_name = name.rsplit(".", 1)[-1]
        is_no_decay = (
            id(param) in norm_param_ids
            or leaf_name in _NO_DECAY_PARAM_NAMES
            or param.ndim <= 1
        )
        if is_no_decay:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return decay_params, no_decay_params


def build_optimizer_and_scheduler(
    model: nn.Module,
    cfg: Dict[str, Any],
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """Construct AdamW with selective weight decay and linear-warmup cosine scheduler."""
    optim_cfg = cfg["optim"]
    lr = float(optim_cfg["lr"])
    min_lr = float(optim_cfg.get("min_lr", 1.0e-6))
    weight_decay = float(optim_cfg.get("weight_decay", 1.0e-4))
    betas = tuple(float(b) for b in optim_cfg.get("betas", (0.9, 0.999)))
    total_steps = int(optim_cfg["total_steps"])
    warmup_steps = int(optim_cfg.get("warmup_steps", 0))

    decay_params, no_decay_params = split_decay_parameters(model)
    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=lr,
        betas=(betas[0], betas[1]),
    )
    lr_lambda = WarmupCosineLambda(
        base_lr=lr,
        min_lr=min_lr,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler
