"""Tests for Stage 2 config loading and validation CLI."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn

from scripts.validate_teacher_dump import main as validate_main
from src.data.teacher_dump import TeacherDumpDataset, collate_teacher_examples
from src.models.student_vlm import StudentVLM
from src.train.config import load_stage2_config
from src.train.stage2 import (
    load_stage2_artifacts,
    prepare_stage2_model_inputs,
    save_stage2_artifacts,
)

MakeDump = Callable[..., tuple[Path, Path]]


def test_stage2_config_loads_defaults() -> None:
    config = load_stage2_config("configs/stage2.yaml")
    assert config.loss.alpha == 1.0
    assert config.loss.beta == 0.1
    assert config.model.lora_rank == 64
    assert config.data.test_split == "data/splits/test.json"
    assert config.model.processor_name == config.model.backbone_name


def test_validation_cli_passes_on_fixture(mini_dump: tuple[Path, Path]) -> None:
    dump_root, split_file = mini_dump
    assert validate_main(["--root", str(dump_root), "--splits", str(split_file)]) == 0


def test_validation_cli_fails_on_corrupt_fixture(make_dump: MakeDump) -> None:
    dump_root, split_file = make_dump(corruption="missing_hidden")
    assert validate_main(["--root", str(dump_root), "--splits", str(split_file)]) == 1


def test_prepare_stage2_model_inputs_without_processor(mini_dump: tuple[Path, Path]) -> None:
    dump_root, split_file = mini_dump
    dataset = TeacherDumpDataset(dump_root, split_file)
    batch = collate_teacher_examples([dataset[0]])
    config = load_stage2_config("configs/stage2.yaml")
    prepared = prepare_stage2_model_inputs(batch, config, torch.device("cpu"), processor=None)
    assert prepared["input_ids"].shape == (1, 3)
    assert prepared["hidden_position_mask"].shape == (1, 3)
    assert prepared["logit_position_mask"].shape == (1, 3)


def test_save_stage2_artifacts_writes_hidden_adapter(tmp_path: Path) -> None:
    model = StudentVLM(
        backbone=nn.Sequential(nn.Linear(2, 2)), student_hidden_dim=2, teacher_hidden_dim=4
    )
    output_dir = tmp_path / "student_vlm"
    save_stage2_artifacts(model, output_dir)
    assert (output_dir / "hidden_adapter.pt").is_file()
    assert (output_dir / "backbone_state.pt").is_file()
    assert (output_dir / "metadata.json").is_file()


def test_load_stage2_artifacts_restores_hidden_adapter(tmp_path: Path) -> None:
    model = StudentVLM(
        backbone=nn.Sequential(nn.Linear(2, 2)), student_hidden_dim=2, teacher_hidden_dim=4
    )
    output_dir = tmp_path / "student_vlm"
    save_stage2_artifacts(model, output_dir)
    with torch.no_grad():
        for parameter in model.hidden_adapter.parameters():
            parameter.zero_()
    load_stage2_artifacts(model, output_dir)
    assert any(
        parameter.detach().abs().sum().item() > 0.0
        for parameter in model.hidden_adapter.parameters()
    )
