from __future__ import annotations

import torch
import torch.nn.functional as F


def _horizon_weights(steps: int, decay: float, device: torch.device) -> torch.Tensor:
    return torch.tensor([decay ** index for index in range(steps)], device=device, dtype=torch.float32)


def trajectory_l2_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    heading_weight: float = 0.1,
    horizon_decay: float = 0.95,
    use_huber: bool = True,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    weights = _horizon_weights(prediction.size(1), horizon_decay, prediction.device).view(1, -1, 1)
    pos_pred, head_pred = prediction[..., :2], prediction[..., 2:]
    pos_tgt, head_tgt = target[..., :2], target[..., 2:]
    if use_huber:
        pos_loss = F.huber_loss(pos_pred, pos_tgt, reduction="none", delta=huber_delta)
        head_loss = F.huber_loss(head_pred, head_tgt, reduction="none", delta=huber_delta)
    else:
        pos_loss = (pos_pred - pos_tgt) ** 2
        head_loss = (head_pred - head_tgt) ** 2
    loss = weights * pos_loss + heading_weight * weights * head_loss
    return loss.mean()


def minade(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.norm(prediction[..., :2] - target[..., :2], dim=-1).mean(dim=-1).mean()
