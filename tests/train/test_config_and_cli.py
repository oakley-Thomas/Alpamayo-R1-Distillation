"""Tests for Stage 2 config loading and validation CLI."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image
from torch import nn

from scripts.validate_teacher_dump import main as validate_main
from src.data.teacher_dump import TeacherDumpDataset, collate_teacher_examples
from src.models.student_vlm import StudentVLM
from src.train.config import load_stage2_config
from src.train.stage2 import (
    load_stage2_artifacts,
    prepare_stage2_model_inputs,
    resolve_stage2_compute_dtype,
    save_stage2_artifacts,
    select_stage2_frame_paths,
)

MakeDump = Callable[..., tuple[Path, Path]]


class FakeTokenizer:
    """Tokenizer stub that exposes deterministic Qwen-token labels."""

    def __init__(self, token_ids: list[int]) -> None:
        self.token_ids = token_ids
        self.seen_text: str | None = None

    def __call__(self, text: str, **_kwargs: object) -> dict[str, torch.Tensor]:
        """Return configured token IDs for any text."""
        self.seen_text = text
        return {"input_ids": torch.tensor([self.token_ids], dtype=torch.long)}


class FakeStage2Processor:
    """Processor stub with separate prompt and CoC tokenization paths."""

    def __init__(self, tokenizer: FakeTokenizer) -> None:
        self.tokenizer = tokenizer

    def apply_chat_template(self, messages: list[dict[str, Any]], **_kwargs: object) -> str:
        """Return a deterministic prompt string."""
        return "prompt"

    def __call__(self, **_kwargs: object) -> dict[str, torch.Tensor]:
        """Return deterministic prompt token IDs."""
        return {
            "input_ids": torch.tensor([[11, 12, 13]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }


def test_stage2_config_loads_defaults() -> None:
    config = load_stage2_config("configs/stage2.yaml")
    assert config.loss.alpha == 1.0
    assert config.loss.beta == 0.1
    assert config.model.lora_rank == 64
    assert config.data.max_frames == 32
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


def test_select_frame_paths_samples_across_clip() -> None:
    paths = [Path(f"frame_{index:04d}.jpg") for index in range(10)]

    selected = select_stage2_frame_paths(paths, max_frames=4)

    assert [path.name for path in selected] == [
        "frame_0000.jpg",
        "frame_0003.jpg",
        "frame_0006.jpg",
        "frame_0009.jpg",
    ]


def test_prepare_stage2_model_inputs_retokenizes_coc_text(
    mini_dump: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dump_root, split_file = mini_dump
    dataset = TeacherDumpDataset(dump_root, split_file)
    batch = collate_teacher_examples([dataset[0]])
    config = load_stage2_config("configs/stage2.yaml")
    tokenizer = FakeTokenizer([101, 102])
    processor = FakeStage2Processor(tokenizer)

    def fake_load_rgb_frame(_path: Path) -> Image.Image:
        return Image.new("RGB", (1, 1))

    monkeypatch.setattr("src.train.stage2._load_rgb_frame", fake_load_rgb_frame)

    prepared = prepare_stage2_model_inputs(batch, config, torch.device("cpu"), processor)

    assert tokenizer.seen_text == "yield to pedestrian"
    assert prepared["input_ids"].tolist() == [[11, 12, 13, 101, 102]]
    assert prepared["lm_token_ids"].tolist() == [[101, 102]]
    assert prepared["lm_token_mask"].tolist() == [[True, True]]
    assert prepared["hidden_position_mask"].tolist() == [[True, True, True, False, False]]
    assert prepared["logit_position_mask"].tolist() == [[False, False, True, True, False]]


def test_stage2_compute_dtype_uses_fp16_when_bf16_disabled() -> None:
    config = load_stage2_config("configs/stage2.yaml")
    config = replace(config, training=replace(config.training, bf16=False))

    assert resolve_stage2_compute_dtype(config) is torch.float16


def test_stage2_compute_dtype_rejects_unsupported_bf16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_stage2_config("configs/stage2.yaml")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    with pytest.raises(RuntimeError, match="does not support bf16"):
        resolve_stage2_compute_dtype(config)


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
