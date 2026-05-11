"""Single-step flow-matching Action Expert for Stage 3."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from src.data.teacher_dump import TRAJECTORY_DIM, WAYPOINTS


@dataclass(frozen=True)
class ActionExpertConfig:
    """Configuration for the Stage 3 flow-matching Action Expert."""

    teacher_hidden_dim: int
    hidden_dim: int = 768
    ffn_dim: int = 3072
    num_layers: int = 8
    num_heads: int = 12
    dropout: float = 0.1
    waypoints: int = WAYPOINTS
    trajectory_dim: int = TRAJECTORY_DIM


class FlowMatchingActionExpert(nn.Module):
    """Transformer decoder that predicts a rectified-flow velocity field."""

    def __init__(self, config: ActionExpertConfig) -> None:
        """Initialize the Action Expert.

        Args:
            config: Architecture dimensions and dropout settings.
        """
        module_init = cast(
            Callable[[nn.Module], None], object.__getattribute__(nn.Module, "__init__")
        )
        module_init(self)
        _validate_config(config)
        self.config = config
        self.noise_projection = nn.Linear(config.trajectory_dim, config.hidden_dim)
        self.waypoint_positions = nn.Parameter(torch.zeros(config.waypoints, config.hidden_dim))
        self.time_mlp = nn.Sequential(
            nn.Linear(1, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.hidden_projection = nn.Linear(config.teacher_hidden_dim, config.hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)
        self.output_head = nn.Linear(config.hidden_dim, config.trajectory_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        hidden_states: torch.Tensor,
        hidden_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the velocity field at interpolation time ``t``.

        Args:
            x_t: Noisy trajectory, shape (B, 64, 3).
            t: Flow-matching interpolation time in [0, 1], shape (B,).
            hidden_states: Frozen Stage 2 conditioning states, shape (B, T_cond, D_h).
            hidden_mask: Optional valid conditioning positions, shape (B, T_cond).

        Returns:
            Predicted velocity field, shape (B, 64, 3).
        """
        _validate_forward_inputs(x_t, t, hidden_states, hidden_mask, self.config)
        query = self.noise_projection(x_t)
        query = query + self.waypoint_positions.unsqueeze(0)
        query = query + self.time_mlp(t.unsqueeze(-1)).unsqueeze(1)
        memory = self.hidden_projection(hidden_states)
        memory_key_padding_mask = None if hidden_mask is None else ~hidden_mask
        decoded = self.decoder(
            tgt=query,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_head(decoded)

    def single_step(
        self,
        noise: torch.Tensor,
        hidden_states: torch.Tensor,
        hidden_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the deployed one-step Euler path from noise to trajectory.

        Args:
            noise: Initial Gaussian trajectory sample, shape (B, 64, 3).
            hidden_states: Frozen Stage 2 conditioning states, shape (B, T_cond, D_h).
            hidden_mask: Optional valid conditioning positions, shape (B, T_cond).

        Returns:
            Single-step trajectory prediction, shape (B, 64, 3).
        """
        t_zero = noise.new_zeros((noise.shape[0],))
        return noise + self.forward(noise, t_zero, hidden_states, hidden_mask)


def _validate_config(config: ActionExpertConfig) -> None:
    if config.teacher_hidden_dim <= 0:
        raise ValueError("teacher_hidden_dim must be positive")
    if config.hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if config.ffn_dim <= 0:
        raise ValueError("ffn_dim must be positive")
    if config.num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if config.num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if config.hidden_dim % config.num_heads != 0:
        raise ValueError("hidden_dim must be divisible by num_heads")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if config.waypoints <= 0:
        raise ValueError("waypoints must be positive")
    if config.trajectory_dim <= 0:
        raise ValueError("trajectory_dim must be positive")


def _validate_forward_inputs(
    x_t: torch.Tensor,
    t: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_mask: torch.Tensor | None,
    config: ActionExpertConfig,
) -> None:
    if x_t.ndim != 3 or x_t.shape[1:] != (config.waypoints, config.trajectory_dim):
        raise ValueError(
            "x_t must have shape "
            f"(B, {config.waypoints}, {config.trajectory_dim}); got {tuple(x_t.shape)}"
        )
    if t.ndim != 1 or t.shape[0] != x_t.shape[0]:
        raise ValueError("t must have shape (B,)")
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape (B, T_cond, D_h)")
    if hidden_states.shape[0] != x_t.shape[0]:
        raise ValueError("hidden_states batch dimension must match x_t")
    if hidden_states.shape[2] != config.teacher_hidden_dim:
        raise ValueError(
            "hidden_states last dimension must match teacher_hidden_dim; "
            f"got {hidden_states.shape[2]} and {config.teacher_hidden_dim}"
        )
    if t.device != x_t.device or hidden_states.device != x_t.device:
        raise ValueError("x_t, t, and hidden_states must be on the same device")
    if torch.any(t < 0) or torch.any(t > 1):
        raise ValueError("t values must be in [0, 1]")
    if hidden_mask is None:
        return
    if hidden_mask.shape != hidden_states.shape[:2]:
        raise ValueError("hidden_mask must have shape (B, T_cond)")
    if hidden_mask.device != x_t.device:
        raise ValueError("hidden_mask must be on the same device as x_t")
    if hidden_mask.dtype is not torch.bool:
        raise ValueError("hidden_mask must be boolean")
    if torch.any(hidden_mask.sum(dim=1) == 0):
        raise ValueError("each sample must have at least one valid hidden state")
