"""Tests for teacher dump validation and loading."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from src.data.teacher_dump import (
    ClipValidationError,
    TeacherDumpDataset,
    collate_teacher_examples,
    validate_teacher_clip,
)

MakeDump = Callable[..., tuple[Path, Path]]


def test_teacher_dump_dataset_loads_two_clip_fixture(mini_dump: tuple[Path, Path]) -> None:
    dump_root, split_file = mini_dump
    dataset = TeacherDumpDataset(dump_root, split_file)
    assert len(dataset) == 2
    example = dataset[0]
    assert example["clip_id"] == "clip-a"
    assert example["coc_text"] == "yield to pedestrian"
    assert example["token_ids"].shape == (3,)
    assert example["top_k_logits"].shape == (3, 32)
    assert example["teacher_hidden_states"].shape == (3, 4)


def test_collate_teacher_examples_pads_batch(mini_dump: tuple[Path, Path]) -> None:
    dump_root, split_file = mini_dump
    dataset = TeacherDumpDataset(dump_root, split_file)
    batch = collate_teacher_examples([dataset[0], dataset[1]])
    assert batch["coc_text"] == ["yield to pedestrian", "yield to pedestrian"]
    assert batch["token_ids"].shape == (2, 3)
    assert batch["teacher_hidden_states"].shape == (2, 3, 4)
    assert batch["token_mask"].all()


def test_validate_teacher_clip_rejects_missing_hidden_states(make_dump: MakeDump) -> None:
    dump_root, _split_file = make_dump(corruption="missing_hidden")
    with pytest.raises(ClipValidationError, match=r"clip-a.*hidden_states"):
        validate_teacher_clip(dump_root / "clip-a")


def test_validate_teacher_clip_rejects_malformed_topk(make_dump: MakeDump) -> None:
    dump_root, _split_file = make_dump(corruption="malformed_topk")
    with pytest.raises(ClipValidationError, match=r"clip-a.*top-k"):
        validate_teacher_clip(dump_root / "clip-a")


def test_missing_kv_cache_is_optional_by_default(make_dump: MakeDump) -> None:
    dump_root, split_file = make_dump(corruption="missing_kv_file")
    dataset = TeacherDumpDataset(dump_root, split_file, include_kv_cache=False)
    assert len(dataset) == 1


def test_include_kv_cache_requires_shards(make_dump: MakeDump) -> None:
    dump_root, split_file = make_dump(corruption="missing_kv_file")
    with pytest.raises(ClipValidationError, match=r"clip-a.*kv_layer_000"):
        TeacherDumpDataset(dump_root, split_file, include_kv_cache=True)


def test_include_kv_cache_accepts_valid_shards(make_dump: MakeDump) -> None:
    dump_root, split_file = make_dump(include_kv=True)
    dataset = TeacherDumpDataset(dump_root, split_file, include_kv_cache=True)
    assert dataset[0]["kv_cache_files"][0].name == "kv_layer_000.npz"
