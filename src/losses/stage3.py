"""Stage 3 rectified-flow losses for the Action Expert."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import torch
from torch.nn import functional as F


class Stage3VelocityModel(Protocol):
    """Callable interface required by the Stage 3 loss."""

    def __call__(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        hidden_states: torch.Tensor,
        hidden_mask: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class Stage3LossConfig:
    """Configuration for Stage 3 loss weighting."""

    gamma: float = 0.5


@dataclass(frozen=True)
class RectifiedFlowSample:
    """Sampled rectified-flow interpolation inputs."""

    x_0: torch.Tensor
    x_1: torch.Tensor
    t: torch.Tensor
    x_t: torch.Tensor
    target_velocity: torch.Tensor


@dataclass(frozen=True)
class Stage3LossOutput:
    """Named Stage 3 loss components."""

    total: torch.Tensor
    flow_matching: torch.Tensor
    trajectory: torch.Tensor


def sample_rectified_flow_inputs(
    trajectories: torch.Tensor,
    generator: torch.Generator | None = None,
) -> RectifiedFlowSample:
    """Sample rectified-flow training inputs from normalized teacher trajectories.

    Args:
        trajectories: Teacher trajectory samples ``x_1``, shape (B, 64, 3).
        generator: Optional torch RNG for deterministic tests or runs.

    Returns:
        Gaussian noise ``x_0``, interpolation time ``t``, interpolated
        trajectory ``x_t``, and target velocity ``x_1 - x_0``.
    """
    if trajectories.ndim != 3:
        raise ValueError("trajectories must have shape (B, 64, 3)")
    x_1 = trajectories
    x_0 = torch.randn(
        trajectories.shape,
        device=trajectories.device,
        dtype=trajectories.dtype,
        generator=generator,
    )
    t = torch.rand(
        (trajectories.shape[0],),
        device=trajectories.device,
        dtype=trajectories.dtype,
        generator=generator,
    )
    t_view = t.reshape(-1, 1, 1)
    x_t: torch.Tensor = (torch.ones_like(t_view) - t_view) * x_0 + t_view * x_1
    target_velocity = x_1 - x_0
    return RectifiedFlowSample(
        x_0=x_0,
        x_1=x_1,
        t=t,
        x_t=x_t,
        target_velocity=target_velocity,
    )


def compute_stage3_loss(
    *,
    model: Stage3VelocityModel,
    batch: Mapping[str, torch.Tensor],
    config: Stage3LossConfig,
    generator: torch.Generator | None = None,
    sample: RectifiedFlowSample | None = None,
) -> Stage3LossOutput:
    """Compute the Stage 3 rectified-flow objective.

    Args:
        model: Action Expert returning velocity predictions, shape (B, 64, 3).
        batch: Tensor batch containing normalized teacher trajectories and
            frozen conditioning hidden states.
        config: Loss weight for direct single-step trajectory regression.
        generator: Optional RNG used when ``sample`` is not provided.
        sample: Optional pre-sampled rectified-flow inputs for deterministic tests.

    Returns:
        Total loss and named flow-matching and trajectory components.

    Raises:
        FloatingPointError: If any component becomes non-finite.
    """
    if config.gamma < 0.0:
        raise ValueError("Stage 3 gamma must be non-negative")
    trajectories = batch["teacher_trajectories"]
    hidden_states = batch["conditioning_hidden_states"]
    hidden_mask = batch.get("hidden_mask")
    flow_sample = sample or sample_rectified_flow_inputs(trajectories, generator=generator)
    _validate_sample(flow_sample, trajectories)

    velocity = model(flow_sample.x_t, flow_sample.t, hidden_states, hidden_mask)
    flow_matching = F.mse_loss(velocity, flow_sample.target_velocity)

    t_zero = flow_sample.x_0.new_zeros((flow_sample.x_0.shape[0],))
    single_step_velocity = model(flow_sample.x_0, t_zero, hidden_states, hidden_mask)
    predicted_x1 = flow_sample.x_0 + single_step_velocity
    trajectory = F.smooth_l1_loss(predicted_x1, flow_sample.x_1)
    total = flow_matching + config.gamma * trajectory

    for name, value in (
        ("flow_matching", flow_matching),
        ("trajectory", trajectory),
        ("total", total),
    ):
        _check_finite(name, value)
    return Stage3LossOutput(total=total, flow_matching=flow_matching, trajectory=trajectory)


def _validate_sample(sample: RectifiedFlowSample, trajectories: torch.Tensor) -> None:
    if sample.x_1.shape != trajectories.shape:
        raise ValueError("rectified-flow sample x_1 must match batch trajectories")
    for name, value in (
        ("x_0", sample.x_0),
        ("x_t", sample.x_t),
        ("target_velocity", sample.target_velocity),
    ):
        if value.shape != trajectories.shape:
            raise ValueError(f"rectified-flow sample {name} must match trajectory shape")
    if sample.t.shape != trajectories.shape[:1]:
        raise ValueError("rectified-flow sample t must have shape (B,)")


def _check_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} became non-finite")
