"""Tests for Stage 3 flow-matching losses."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest
import torch
from torch import nn

from src.losses.stage3 import (
    RectifiedFlowSample,
    Stage3LossConfig,
    compute_stage3_loss,
    sample_rectified_flow_inputs,
)


class ConstantVelocityModel(nn.Module):
    """Model stub returning a configured velocity tensor."""

    def __init__(self, velocity: torch.Tensor) -> None:
        module_init = cast(
            Callable[[nn.Module], None], object.__getattribute__(nn.Module, "__init__")
        )
        module_init(self)
        self.velocity = velocity

    def forward(
        self,
        x_t: torch.Tensor,
        _t: torch.Tensor,
        _hidden_states: torch.Tensor,
        _hidden_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the configured velocity with the requested dtype/device."""
        return self.velocity.to(device=x_t.device, dtype=x_t.dtype)


def _batch() -> dict[str, torch.Tensor]:
    return {
        "teacher_trajectories": torch.ones((2, 64, 3)),
        "student_hidden_states": torch.randn(2, 3, 4),
        "hidden_mask": torch.ones((2, 3), dtype=torch.bool),
    }


def test_sample_rectified_flow_inputs_is_finite() -> None:
    sample = sample_rectified_flow_inputs(torch.zeros((2, 64, 3)))

    assert torch.isfinite(sample.x_t).all()
    assert torch.isfinite(sample.target_velocity).all()
    assert sample.t.shape == (2,)


def test_stage3_loss_is_zero_for_exact_velocity() -> None:
    batch = _batch()
    x_0 = torch.zeros((2, 64, 3))
    x_1 = batch["teacher_trajectories"]
    target_velocity = x_1 - x_0
    sample = RectifiedFlowSample(
        x_0=x_0,
        x_1=x_1,
        t=torch.full((2,), 0.25),
        x_t=torch.full((2, 64, 3), 0.25),
        target_velocity=target_velocity,
    )

    loss = compute_stage3_loss(
        model=ConstantVelocityModel(target_velocity),
        batch=batch,
        config=Stage3LossConfig(),
        sample=sample,
    )

    assert abs(loss.total.item()) < 1e-6
    assert abs(loss.flow_matching.item()) < 1e-6
    assert abs(loss.trajectory.item()) < 1e-6


def test_stage3_loss_applies_gamma_weight() -> None:
    batch = _batch()
    x_0 = torch.zeros((2, 64, 3))
    x_1 = batch["teacher_trajectories"]
    sample = RectifiedFlowSample(
        x_0=x_0,
        x_1=x_1,
        t=torch.zeros((2,)),
        x_t=x_0,
        target_velocity=x_1 - x_0,
    )

    loss = compute_stage3_loss(
        model=ConstantVelocityModel(torch.zeros((2, 64, 3))),
        batch=batch,
        config=Stage3LossConfig(gamma=0.25),
        sample=sample,
    )

    assert abs(loss.flow_matching.item() - 1.0) < 1e-6
    assert abs(loss.trajectory.item() - 0.5) < 1e-6
    assert abs(loss.total.item() - 1.125) < 1e-6


def test_stage3_loss_rejects_nan() -> None:
    batch = _batch()
    x_0 = torch.zeros((2, 64, 3))
    x_1 = batch["teacher_trajectories"]
    sample = RectifiedFlowSample(
        x_0=x_0,
        x_1=x_1,
        t=torch.zeros((2,)),
        x_t=x_0,
        target_velocity=x_1 - x_0,
    )

    with pytest.raises(FloatingPointError, match="non-finite"):
        compute_stage3_loss(
            model=ConstantVelocityModel(torch.full((2, 64, 3), float("nan"))),
            batch=batch,
            config=Stage3LossConfig(),
            sample=sample,
        )
