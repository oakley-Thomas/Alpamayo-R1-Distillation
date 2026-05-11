"""Tests for Stage 2 loss components."""

from __future__ import annotations

import pytest
import torch

from src.losses.stage2 import Stage2LossConfig, Stage2ModelOutput, compute_stage2_loss


def _batch() -> dict[str, torch.Tensor]:
    lm_token_ids = torch.tensor([[3, 4]])
    top_k_token_ids = torch.arange(32).repeat(1, 2, 1)
    top_k_token_ids[0, 0, 0] = 155678
    top_k_token_ids[0, 1, 0] = 155681
    top_k_logits = torch.full((1, 2, 32), -20.0)
    top_k_logits[:, :, 0] = 20.0
    teacher_hidden = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    return {
        "lm_token_ids": lm_token_ids,
        "top_k_token_ids": top_k_token_ids,
        "top_k_logits": top_k_logits,
        "teacher_hidden_states": teacher_hidden,
        "lm_token_mask": torch.ones((1, 2), dtype=torch.bool),
        "hidden_mask": torch.ones((1, 2), dtype=torch.bool),
    }


def test_coc_kl_is_disabled_for_teacher_token_vocab_mismatch() -> None:
    batch = _batch()
    outputs = Stage2ModelOutput(
        logits=torch.zeros((1, 2, 64)),
        adapted_hidden_states=batch["teacher_hidden_states"].clone(),
    )
    loss = compute_stage2_loss(batch, outputs, Stage2LossConfig())
    assert loss.coc_kl.item() == 0.0
    assert torch.isfinite(loss.total)


def test_hidden_alignment_is_zero_for_equal_tensors() -> None:
    batch = _batch()
    outputs = Stage2ModelOutput(
        logits=torch.zeros((1, 2, 64)),
        adapted_hidden_states=batch["teacher_hidden_states"].clone(),
    )
    loss = compute_stage2_loss(batch, outputs, Stage2LossConfig())
    assert abs(loss.hidden_align.item()) < 1e-6


def test_combined_loss_exposes_components() -> None:
    batch = _batch()
    outputs = Stage2ModelOutput(
        logits=torch.zeros((1, 2, 64)),
        adapted_hidden_states=batch["teacher_hidden_states"].clone(),
    )
    loss = compute_stage2_loss(batch, outputs, Stage2LossConfig(alpha=0.3, beta=0.2))
    assert loss.total.ndim == 0
    assert loss.coc_kl.ndim == 0
    assert loss.hidden_align.ndim == 0
    assert loss.lm_ce.ndim == 0


def test_combined_loss_rejects_nan() -> None:
    batch = _batch()
    logits = torch.zeros((1, 2, 64))
    logits[0, 0, 0] = float("nan")
    outputs = Stage2ModelOutput(
        logits=logits,
        adapted_hidden_states=batch["teacher_hidden_states"].clone(),
    )
    with pytest.raises(FloatingPointError, match="non-finite"):
        compute_stage2_loss(batch, outputs, Stage2LossConfig())
