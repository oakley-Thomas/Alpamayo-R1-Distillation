"""Tests for the Stage 3 flow-matching Action Expert."""

from __future__ import annotations

import pytest
import torch

from src.models.action_expert import ActionExpertConfig, FlowMatchingActionExpert


def _small_model() -> FlowMatchingActionExpert:
    return FlowMatchingActionExpert(
        ActionExpertConfig(
            teacher_hidden_dim=4,
            hidden_dim=12,
            ffn_dim=24,
            num_layers=2,
            num_heads=3,
            dropout=0.0,
        )
    )


def test_action_expert_forward_shape() -> None:
    model = _small_model()
    x_t = torch.randn(2, 64, 3)
    t = torch.tensor([0.1, 0.9])
    hidden_states = torch.randn(2, 5, 4)

    velocity = model(x_t, t, hidden_states)

    assert velocity.shape == (2, 64, 3)


def test_action_expert_accepts_masked_conditioning() -> None:
    model = _small_model()
    x_t = torch.randn(2, 64, 3)
    t = torch.tensor([0.2, 0.4])
    hidden_states = torch.randn(2, 5, 4)
    hidden_mask = torch.tensor(
        [[True, True, False, False, False], [True, False, True, False, False]]
    )

    velocity = model(x_t, t, hidden_states, hidden_mask)

    assert velocity.shape == (2, 64, 3)


def test_action_expert_single_step_shape() -> None:
    model = _small_model()
    noise = torch.randn(2, 64, 3)
    hidden_states = torch.randn(2, 5, 4)

    prediction = model.single_step(noise, hidden_states)

    assert prediction.shape == (2, 64, 3)


def test_action_expert_rejects_invalid_shapes() -> None:
    model = _small_model()
    x_t = torch.randn(2, 63, 3)
    t = torch.tensor([0.2, 0.4])
    hidden_states = torch.randn(2, 5, 4)

    with pytest.raises(ValueError, match="x_t must have shape"):
        model(x_t, t, hidden_states)


def test_action_expert_rejects_empty_hidden_mask() -> None:
    model = _small_model()
    x_t = torch.randn(2, 64, 3)
    t = torch.tensor([0.2, 0.4])
    hidden_states = torch.randn(2, 5, 4)
    hidden_mask = torch.zeros((2, 5), dtype=torch.bool)

    with pytest.raises(ValueError, match="at least one valid hidden state"):
        model(x_t, t, hidden_states, hidden_mask)
