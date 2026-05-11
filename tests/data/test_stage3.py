"""Tests for Stage 3 trajectory dataset utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.stage3 import (
    Stage3TrajectoryDataset,
    TrajectoryNormStats,
    collate_stage3_examples,
    compute_trajectory_norm_stats,
)
from src.data.teacher_dump import ClipValidationError

MakeDump = Callable[..., tuple[Path, Path]]


def _write_hidden_cache(cache_dir: Path, clip_ids: Iterable[str]) -> None:
    cache_dir.mkdir()
    hidden_states = np.arange(12, dtype=np.float16).reshape(3, 4)
    for clip_id in clip_ids:
        np.save(cache_dir / f"{clip_id}.npy", hidden_states)


def test_stage3_dataset_loads_hidden_cache(mini_dump: tuple[Path, Path], tmp_path: Path) -> None:
    dump_root, split_file = mini_dump
    cache_dir = tmp_path / "hidden_cache"
    _write_hidden_cache(cache_dir, ["clip-a", "clip-b"])

    dataset = Stage3TrajectoryDataset(dump_root, split_file, cache_dir)
    example = dataset[0]

    assert len(dataset) == 2
    assert dataset.hidden_dim == 4
    assert example["clip_id"] == "clip-a"
    assert example["teacher_trajectories"].shape == (16, 64, 3)
    assert example["student_hidden_states"].shape == (3, 4)
    assert torch.equal(example["hidden_mask"], torch.ones(3, dtype=torch.bool))


def test_stage3_dataset_rejects_missing_hidden_cache(
    make_dump: MakeDump,
    tmp_path: Path,
) -> None:
    dump_root, split_file = make_dump()
    cache_dir = tmp_path / "hidden_cache"
    cache_dir.mkdir()

    with pytest.raises(ClipValidationError, match=r"clip-a.*student hidden cache"):
        Stage3TrajectoryDataset(dump_root, split_file, cache_dir)


def test_trajectory_norm_stats_round_trip(tmp_path: Path) -> None:
    stats = TrajectoryNormStats(mean=(1.0, 2.0, 3.0), std=(2.0, 4.0, 5.0))
    trajectories = torch.tensor([[[3.0, 10.0, 8.0], [1.0, 2.0, 3.0]]])
    stats_path = tmp_path / "traj_norm_stats.json"

    normalized = stats.normalize_tensor(trajectories)
    restored = stats.denormalize_tensor(normalized)
    stats.save(stats_path)

    assert torch.allclose(restored, trajectories)
    assert TrajectoryNormStats.load(stats_path) == stats


def test_compute_trajectory_norm_stats_uses_train_split(
    mini_dump: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    dump_root, split_file = mini_dump
    cache_dir = tmp_path / "hidden_cache"
    _write_hidden_cache(cache_dir, ["clip-a", "clip-b"])
    dataset = Stage3TrajectoryDataset(dump_root, split_file, cache_dir)

    stats = compute_trajectory_norm_stats(dataset)

    assert stats.mean == (0.0, 0.0, 0.0)
    assert all(value > 0.0 for value in stats.std)


def test_collate_stage3_examples_repeats_hidden_states(
    mini_dump: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    dump_root, split_file = mini_dump
    cache_dir = tmp_path / "hidden_cache"
    _write_hidden_cache(cache_dir, ["clip-a", "clip-b"])
    dataset = Stage3TrajectoryDataset(dump_root, split_file, cache_dir)

    batch = collate_stage3_examples([dataset[0]])

    assert batch["clip_id"] == ["clip-a"] * 16
    assert batch["teacher_trajectories"].shape == (16, 64, 3)
    assert batch["student_hidden_states"].shape == (16, 3, 4)
    assert batch["hidden_mask"].shape == (16, 3)
