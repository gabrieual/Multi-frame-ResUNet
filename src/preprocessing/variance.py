"""Temporal uncertainty map U_var highlighting turbulent oscillation strength."""

import torch


@torch.no_grad()
def compute_temporal_std(frames: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes temporal uncertainty map U_var as defined in ResUNetMultiFrame.md §2.3:
        U_var = (1/3) * sum_c sqrt( mean_t((I_{t,c} - I_{mean,c})^2) + 1e-6 )
    Always computed in float32 for numerical stability under AMP.

    Args:
        frames: [B, N, 3, H, W]
    Returns:
        u_var: [B, 1, H, W]
    """
    orig_dtype = frames.dtype
    frames_f32 = frames.float()
    mean_f32 = frames_f32.mean(dim=1, keepdim=True)
    var_c = (frames_f32 - mean_f32).square().mean(dim=1)  # [B, 3, H, W]
    std_c = torch.sqrt(var_c + eps)
    u_var = std_c.mean(dim=1, keepdim=True)               # [B, 1, H, W]
    return u_var.to(orig_dtype)
