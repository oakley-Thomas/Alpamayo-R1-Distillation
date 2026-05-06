"""Tests for the Stage 2 student VLM wrapper."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from src.models.student_vlm import HiddenStateAdapter, StudentVLM


class FakeBackbone(nn.Module):
    """Small backbone that mimics a Hugging Face VLM output."""

    def __init__(self) -> None:
        from typing import cast

        module_init = cast(
            Callable[[nn.Module], None], object.__getattribute__(nn.Module, "__init__")
        )
        module_init(self)
        self.embedding = nn.Embedding(16, 6)
        self.lm_head = nn.Linear(6, 16)

    def forward(self, input_ids: torch.Tensor, **_kwargs: Any) -> SimpleNamespace:
        """Return logits and hidden states for fake token IDs."""
        hidden = self.embedding(input_ids)
        return SimpleNamespace(logits=self.lm_head(hidden), hidden_states=(hidden,))


def test_hidden_adapter_shape() -> None:
    adapter = HiddenStateAdapter(student_hidden_dim=6, teacher_hidden_dim=4)
    x = torch.randn(2, 3, 6)
    y = adapter(x)
    assert y.shape == (2, 3, 4)


def test_student_vlm_forward_with_fake_backbone() -> None:
    model = StudentVLM(backbone=FakeBackbone(), student_hidden_dim=6, teacher_hidden_dim=4)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    hidden_position_mask = torch.tensor([[True, False, True], [False, True, True]])
    output = model(input_ids=input_ids, hidden_position_mask=hidden_position_mask)
    assert output.logits.shape == (2, 3, 16)
    assert output.adapted_hidden_states.shape == (2, 2, 4)


def test_student_vlm_selects_logit_positions() -> None:
    model = StudentVLM(backbone=FakeBackbone(), student_hidden_dim=6, teacher_hidden_dim=4)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    logit_position_mask = torch.tensor([[False, True, True, False]])
    output = model(input_ids=input_ids, logit_position_mask=logit_position_mask)
    assert output.logits.shape == (1, 2, 16)
    assert output.adapted_hidden_states.shape == (1, 4, 4)
