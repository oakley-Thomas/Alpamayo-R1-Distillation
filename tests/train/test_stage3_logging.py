"""Tests for Stage 3 training loss logging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
import torch

from src.train.stage3 import run_stage3_training, stage3_loss_log_path


def test_run_stage3_training_writes_per_step_loss_log(
    mini_dump: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dump_root, split_file = mini_dump
    output_root = tmp_path / "stage3"
    config_path = tmp_path / "stage3.yaml"
    _write_stage3_smoke_config(config_path, dump_root, split_file, output_root)

    def fake_run_text_command(_args: list[str]) -> str:
        return "ok"

    monkeypatch.setattr("src.train.stage3._run_text_command", fake_run_text_command)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    run_stage3_training(config_path)

    log_path = stage3_loss_log_path(output_root)
    rows = [
        cast(dict[str, object], json.loads(line))
        for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2
    assert rows[0]["global_step"] == 1
    assert rows[0]["total_steps"] == 2
    for row in rows:
        assert isinstance(row["loss_total"], (int, float))
        assert isinstance(row["loss_flow_matching"], (int, float))
        assert isinstance(row["loss_trajectory"], (int, float))
    first_total = cast(float, rows[0]["loss_total"])
    first_flow_matching = cast(float, rows[0]["loss_flow_matching"])
    first_trajectory = cast(float, rows[0]["loss_trajectory"])
    assert abs(first_total - (first_flow_matching + 0.5 * first_trajectory)) < 1e-6

    stdout = capsys.readouterr().out
    assert "flow_matching=" in stdout
    assert "trajectory=" in stdout


def _write_stage3_smoke_config(
    config_path: Path,
    dump_root: Path,
    split_file: Path,
    output_root: Path,
) -> None:
    config_path.write_text(
        "\n".join(
            [
                "data:",
                f"  teacher_dump_root: {_yaml_string(dump_root)}",
                f"  train_split: {_yaml_string(split_file)}",
                f"  val_split: {_yaml_string(split_file)}",
                f"  test_split: {_yaml_string(split_file)}",
                "  conditioning_source: teacher_dump",
                "  hidden_cache_dir: null",
                "model:",
                "  teacher_hidden_dim: null",
                "  hidden_dim: 12",
                "  ffn_dim: 24",
                "  num_layers: 1",
                "  num_heads: 3",
                "  dropout: 0.0",
                "loss:",
                "  gamma: 0.5",
                "optimizer:",
                "  lr: 0.001",
                "  weight_decay: 0.0",
                "  warmup_fraction: 0.0",
                "training:",
                "  epochs: 1",
                "  batch_size_clips: 1",
                "  bf16: false",
                "  require_cuda: false",
                "  seed: 42",
                "outputs:",
                f"  root: {_yaml_string(output_root)}",
                f"  checkpoint_path: {_yaml_string(output_root / 'action_expert.pt')}",
                f"  norm_stats_path: {_yaml_string(output_root / 'traj_norm_stats.json')}",
                f"  val_predictions_path: {_yaml_string(output_root / 'val_predictions.npz')}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _yaml_string(path: Path) -> str:
    return json.dumps(str(path))
