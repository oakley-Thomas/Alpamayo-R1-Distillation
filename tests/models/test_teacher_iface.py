"""Tests for teacher-conditioning replay helpers."""

from __future__ import annotations

import pytest
import torch

from src.models.teacher_iface import build_stage2_replay_masks


def test_stage2_replay_masks_select_teacher_prefix_and_generated_logits() -> None:
    masks = build_stage2_replay_masks(
        sequence_length=8,
        token_count=3,
        teacher_hidden_length=6,
        conditioning_meta={
            "traj_future_start_offset": 6,
            "prefill_seq_len": 4,
        },
    )
    assert masks.hidden_position_mask.equal(
        torch.tensor([True, True, True, True, True, True, False, False])
    )
    assert masks.logit_position_mask.equal(
        torch.tensor([False, False, False, True, True, True, False, False])
    )


def test_stage2_replay_masks_reject_short_replay() -> None:
    with pytest.raises(ValueError, match="shorter than the teacher conditioning offset"):
        build_stage2_replay_masks(
            sequence_length=3,
            token_count=3,
            teacher_hidden_length=6,
            conditioning_meta={
                "traj_future_start_offset": 6,
                "prefill_seq_len": 4,
            },
        )
