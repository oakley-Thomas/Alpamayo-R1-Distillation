from __future__ import annotations

import torch


def ade(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.norm(prediction[..., :2] - target[..., :2], dim=-1).mean()


def fde(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.norm(prediction[:, -1, :2] - target[:, -1, :2], dim=-1).mean()


def rmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(((prediction - target) ** 2).mean())


def compute_open_loop_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {
        "minADE": float(ade(prediction, target).item()),
        "minFDE": float(fde(prediction, target).item()),
        "RMSE": float(rmse(prediction, target).item()),
    }
