"""Tests for Stage 3 checkpoint serialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from src.data.stage3 import TrajectoryNormStats
from src.models.action_expert import ActionExpertConfig, FlowMatchingActionExpert
from src.train.config import Stage3Config, load_stage3_config
from src.train.stage3 import (
    build_stage3_scheduler,
    load_stage3_checkpoint,
    save_stage3_checkpoint,
)


def _small_model() -> FlowMatchingActionExpert:
    return FlowMatchingActionExpert(
        ActionExpertConfig(
            teacher_hidden_dim=4,
            hidden_dim=12,
            ffn_dim=24,
            num_layers=2,
            num_heads=3,
            dropout=0.0,
        )
    )


def _checkpoint_parts() -> tuple[
    FlowMatchingActionExpert,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    Stage3Config,
    TrajectoryNormStats,
]:
    model = _small_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = build_stage3_scheduler(
        optimizer,
        total_steps=4,
        warmup_fraction=0.25,
    )
    config = load_stage3_config("configs/stage3.yaml")
    norm_stats = TrajectoryNormStats(mean=(1.0, 2.0, 3.0), std=(2.0, 3.0, 4.0))
    return model, optimizer, scheduler, config, norm_stats


def test_save_stage3_checkpoint_is_weights_only_loadable(tmp_path: Path) -> None:
    model, optimizer, scheduler, config, norm_stats = _checkpoint_parts()
    checkpoint_path = tmp_path / "action_expert.pt"

    save_stage3_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        config_path=Path("configs/stage3.yaml"),
        norm_stats=norm_stats,
        checkpoint_path=checkpoint_path,
    )

    checkpoint_obj = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    assert isinstance(checkpoint_obj, dict)
    checkpoint = cast(dict[str, Any], checkpoint_obj)
    rng_state_obj = checkpoint["rng_state"]
    assert isinstance(rng_state_obj, dict)
    rng_state = cast(dict[str, object], rng_state_obj)
    numpy_state_obj = rng_state["numpy"]
    python_state_obj = rng_state["python"]
    assert isinstance(numpy_state_obj, dict)
    assert isinstance(python_state_obj, dict)
    numpy_state = cast(dict[str, object], numpy_state_obj)
    python_state = cast(dict[str, object], python_state_obj)
    assert isinstance(numpy_state["keys"], list)
    assert isinstance(python_state["state"], list)


def test_load_stage3_checkpoint_restores_model_and_stats(tmp_path: Path) -> None:
    model, optimizer, scheduler, config, norm_stats = _checkpoint_parts()
    checkpoint_path = tmp_path / "action_expert.pt"
    expected_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    save_stage3_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        config_path=Path("configs/stage3.yaml"),
        norm_stats=norm_stats,
        checkpoint_path=checkpoint_path,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    loaded_stats = load_stage3_checkpoint(model, checkpoint_path)

    assert loaded_stats == norm_stats
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, expected_state[name])


def test_load_stage3_checkpoint_accepts_legacy_numpy_rng_state(tmp_path: Path) -> None:
    model, _optimizer, _scheduler, _config, norm_stats = _checkpoint_parts()
    checkpoint_path = tmp_path / "legacy_action_expert.pt"
    torch.save(
        {
            "format": "stage3_action_expert",
            "model_state": model.state_dict(),
            "norm_stats": norm_stats.to_json_dict(),
            "rng_state": {"numpy": np.random.get_state()},
        },
        checkpoint_path,
    )

    with pytest.warns(UserWarning, match="legacy Stage 3 checkpoint"):
        loaded_stats = load_stage3_checkpoint(model, checkpoint_path)

    assert loaded_stats == norm_stats
