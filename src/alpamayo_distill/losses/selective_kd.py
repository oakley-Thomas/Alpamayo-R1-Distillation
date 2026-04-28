from __future__ import annotations

import torch


def apply_selective_weights(token_loss: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    if weights is None:
        return token_loss.mean()
    return (token_loss * weights).sum() / weights.sum().clamp_min(1.0)
