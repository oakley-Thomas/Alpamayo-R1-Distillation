"""Tests for teacher dump export helper functions."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.data.export_teacher_dump import (
    TeacherDumpExportError,
    conditioning_record_to_json,
    ensure_denoising_available,
    extract_primary_top_k,
    select_hidden_states_until_offset,
)
from src.models.teacher_iface import AlpamayoConditioningRecord


def test_extract_primary_top_k_preserves_ordered_logits() -> None:
    generated = torch.tensor([[2, 3]])
    logits = (
        torch.tensor([[0.0, 10.0, 20.0, 5.0]]),
        torch.tensor([[8.0, 1.0, 7.0, 6.0]]),
    )

    trace = extract_primary_top_k(generated, logits, top_k=2)

    assert trace.token_ids == [2, 3]
    assert trace.top_k_token_ids == [[2, 1], [0, 2]]
    assert trace.top_k_logits == [[20.0, 10.0], [8.0, 7.0]]


def test_select_hidden_states_until_expert_offset_returns_fp16() -> None:
    hidden = torch.arange(15, dtype=torch.float32).reshape(1, 5, 3)

    selected = select_hidden_states_until_offset(hidden, offset=3)

    assert selected.shape == (3, 3)
    assert selected.dtype == np.float16
    assert selected[-1].tolist() == [6.0, 7.0, 8.0]


def test_conditioning_metadata_json_round_trip() -> None:
    record = AlpamayoConditioningRecord(
        traj_future_start_offset=4,
        prefill_seq_len=8,
        rope_deltas=(1, 2, 3),
        attention_mask_shape=(1, 1, 64, 72),
        generated_seq_len=12,
        n_diffusion_tokens=64,
        kv_cache_files=("kv_layer_000.npz",),
    )

    encoded = json.loads(json.dumps(conditioning_record_to_json(record)))

    assert encoded["traj_future_start_offset"] == 4
    assert encoded["rope_deltas"] == [1, 2, 3]
    assert encoded["kv_cache_files"] == ["kv_layer_000.npz"]


def test_stage3_ready_export_requires_denoising_capture() -> None:
    with pytest.raises(TeacherDumpExportError, match="require --capture-denoising"):
        ensure_denoising_available(capture_denoising=False, require_stage3_fields=True)


def test_stage2_only_export_can_skip_denoising_capture() -> None:
    ensure_denoising_available(capture_denoising=False, require_stage3_fields=False)
