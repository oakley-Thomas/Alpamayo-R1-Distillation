"""Tests for Stage 2 config loading and validation CLI."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from scripts.validate_teacher_dump import main as validate_main
from src.train.config import load_stage2_config

MakeDump = Callable[..., tuple[Path, Path]]


def test_stage2_config_loads_defaults() -> None:
    config = load_stage2_config("configs/stage2.yaml")
    assert config.loss.alpha == 1.0
    assert config.loss.beta == 0.1
    assert config.model.lora_rank == 64


def test_validation_cli_passes_on_fixture(mini_dump: tuple[Path, Path]) -> None:
    dump_root, split_file = mini_dump
    assert validate_main(["--root", str(dump_root), "--splits", str(split_file)]) == 0


def test_validation_cli_fails_on_corrupt_fixture(make_dump: MakeDump) -> None:
    dump_root, split_file = make_dump(corruption="missing_hidden")
    assert validate_main(["--root", str(dump_root), "--splits", str(split_file)]) == 1
