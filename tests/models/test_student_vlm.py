"""Tests for the Stage 2 student VLM wrapper."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from src.models.student_vlm import HiddenStateAdapter, StudentVLM


class FakeBackbone(nn.Module):
    """Small backbone that mimics a Hugging Face VLM output."""

    def __init__(self) -> None:
        super().__init__()
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
