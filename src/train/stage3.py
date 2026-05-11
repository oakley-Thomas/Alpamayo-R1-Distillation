"""Stage 3 training loop for the flow-matching Action Expert."""

from __future__ import annotations

import math
import random
import subprocess
import sys
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.data.stage3 import (
    Stage3TrajectoryDataset,
    TrajectoryNormStats,
    collate_stage3_examples,
    compute_trajectory_norm_stats,
)
from src.losses.stage3 import Stage3LossConfig, compute_stage3_loss
from src.models.action_expert import ActionExpertConfig, FlowMatchingActionExpert
from src.train.config import Stage3Config, load_stage3_config
from src.utils.seed import set_seed


def _run_text_command(args: list[str]) -> str:
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def write_stage3_reproducibility_files(
    config: Stage3Config,
    config_path: Path,
    output_dir: Path,
) -> None:
    """Write config, git SHA, and environment metadata for a Stage 3 run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False)
    git_sha = _run_text_command(["git", "rev-parse", "HEAD"])
    (output_dir / "git_sha.txt").write_text(git_sha + "\n", encoding="utf-8")
    env_text = _run_text_command([sys.executable, "-m", "pip", "freeze"])
    (output_dir / "env.txt").write_text(env_text + "\n", encoding="utf-8")
    (output_dir / "source_config_path.txt").write_text(str(config_path) + "\n", encoding="utf-8")


def resolve_stage3_device(config: Stage3Config) -> torch.device:
    """Resolve the Stage 3 training device from config and hardware."""
    if config.training.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("Stage 3 config requires CUDA, but torch.cuda.is_available() is false")
    if torch.cuda.is_available():
        if config.training.bf16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "Stage 3 config requests bf16, but this CUDA device does not support bf16."
            )
        return torch.device("cuda")
    return torch.device("cpu")


def build_stage3_model(config: Stage3Config, teacher_hidden_dim: int) -> FlowMatchingActionExpert:
    """Build the Stage 3 flow-matching Action Expert from config."""
    configured_dim = config.model.teacher_hidden_dim
    if configured_dim is not None and configured_dim != teacher_hidden_dim:
        raise ValueError(
            "Stage 3 config teacher_hidden_dim does not match hidden cache; "
            f"got {configured_dim} and {teacher_hidden_dim}"
        )
    return FlowMatchingActionExpert(
        ActionExpertConfig(
            teacher_hidden_dim=teacher_hidden_dim,
            hidden_dim=config.model.hidden_dim,
            ffn_dim=config.model.ffn_dim,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout,
        )
    )


def build_stage3_optimizer(
    model: nn.Module,
    config: Stage3Config,
) -> torch.optim.Optimizer:
    """Build the Stage 3 AdamW optimizer."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
    )


def build_stage3_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_fraction: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a cosine schedule with fractional warmup."""
    if total_steps <= 0:
        raise ValueError("Stage 3 scheduler total_steps must be positive")
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("Stage 3 warmup_fraction must be in [0, 1)")

    warmup_steps = int(total_steps * warmup_fraction)

    def lr_lambda(step_idx: int) -> float:
        current_step = step_idx + 1
        if warmup_steps > 0 and current_step <= warmup_steps:
            return current_step / warmup_steps
        cosine_steps = max(total_steps - warmup_steps, 1)
        progress = min((current_step - warmup_steps) / cosine_steps, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_stage3_checkpoint(
    *,
    model: FlowMatchingActionExpert,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: Stage3Config,
    config_path: Path,
    norm_stats: TrajectoryNormStats,
    checkpoint_path: str | Path,
) -> None:
    """Save a resumable Stage 3 checkpoint."""
    destination = Path(checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {
        "format": "stage3_action_expert",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config": asdict(config),
        "source_config_path": str(config_path),
        "norm_stats": norm_stats.to_json_dict(),
        "rng_state": _rng_state(),
    }
    torch.save(checkpoint, destination)


def load_stage3_checkpoint(
    model: FlowMatchingActionExpert,
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> TrajectoryNormStats:
    """Load Action Expert weights and return saved trajectory stats."""
    source = Path(checkpoint_path)
    checkpoint_obj = torch.load(source, map_location=map_location)
    if not isinstance(checkpoint_obj, dict):
        raise RuntimeError(f"{source} did not contain a Stage 3 checkpoint")
    checkpoint = cast(dict[str, object], checkpoint_obj)
    state_obj = checkpoint.get("model_state")
    if not isinstance(state_obj, dict):
        raise RuntimeError(f"{source} missing model_state")
    model.load_state_dict(cast(dict[str, torch.Tensor], state_obj))
    return TrajectoryNormStats.from_json_dict(checkpoint.get("norm_stats"))


def run_stage3_training(config_path: str | Path) -> None:
    """Run Stage 3 flow-matching Action Expert training."""
    config_file = Path(config_path)
    config = load_stage3_config(config_file)
    set_seed(config.training.seed)
    device = resolve_stage3_device(config)

    raw_train_dataset = Stage3TrajectoryDataset(
        teacher_dump_root=config.data.teacher_dump_root,
        split_file=config.data.train_split,
        hidden_cache_dir=config.data.hidden_cache_dir,
    )
    norm_stats = compute_trajectory_norm_stats(raw_train_dataset)
    norm_stats.save(config.outputs.norm_stats_path)
    train_dataset = Stage3TrajectoryDataset(
        teacher_dump_root=config.data.teacher_dump_root,
        split_file=config.data.train_split,
        hidden_cache_dir=config.data.hidden_cache_dir,
        norm_stats=norm_stats,
    )
    val_dataset = Stage3TrajectoryDataset(
        teacher_dump_root=config.data.teacher_dump_root,
        split_file=config.data.val_split,
        hidden_cache_dir=config.data.hidden_cache_dir,
        norm_stats=norm_stats,
    )

    model = build_stage3_model(config, teacher_hidden_dim=train_dataset.hidden_dim).to(device)
    optimizer = build_stage3_optimizer(model, config)
    loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size_clips,
        shuffle=True,
        collate_fn=collate_stage3_examples,
    )
    total_steps = len(loader) * config.training.epochs
    scheduler = build_stage3_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_fraction=config.optimizer.warmup_fraction,
    )
    loss_config = Stage3LossConfig(gamma=config.loss.gamma)

    write_stage3_reproducibility_files(config, config_file, Path(config.outputs.root))
    model.train()
    for _epoch in range(config.training.epochs):
        for batch in loader:
            tensor_batch = _move_stage3_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with _stage3_autocast(config, device):
                loss_output = compute_stage3_loss(
                    model=model,
                    batch=tensor_batch,
                    config=loss_config,
                )
            torch.autograd.backward([loss_output.total])
            optimizer.step()
            scheduler.step()

    save_stage3_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        config_path=config_file,
        norm_stats=norm_stats,
        checkpoint_path=config.outputs.checkpoint_path,
    )
    _write_stage3_predictions(
        model=model,
        dataset=val_dataset,
        norm_stats=norm_stats,
        device=device,
        output_path=config.outputs.val_predictions_path,
        seed=config.training.seed,
    )


def _rng_state() -> dict[str, object]:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _move_stage3_batch(
    batch: Mapping[str, object], device: torch.device
) -> dict[str, torch.Tensor]:
    tensor_batch: dict[str, torch.Tensor] = {}
    for key in ("teacher_trajectories", "student_hidden_states", "hidden_mask"):
        value = batch.get(key)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Stage 3 batch field {key} must be a tensor")
        tensor_batch[key] = value.to(device)
    return tensor_batch


@contextmanager
def _stage3_autocast(config: Stage3Config, device: torch.device) -> Generator[None, None, None]:
    if config.training.bf16:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            yield
    else:
        yield


def _write_stage3_predictions(
    *,
    model: FlowMatchingActionExpert,
    dataset: Stage3TrajectoryDataset,
    norm_stats: TrajectoryNormStats,
    device: torch.device,
    output_path: str | Path,
    seed: int,
) -> None:
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_stage3_examples)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    clip_ids: list[str] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            tensor_batch = _move_stage3_batch(batch, device)
            teacher = tensor_batch["teacher_trajectories"]
            noise = torch.randn(
                teacher.shape,
                device=teacher.device,
                dtype=teacher.dtype,
                generator=generator,
            )
            predicted = model.single_step(
                noise,
                tensor_batch["student_hidden_states"],
                tensor_batch["hidden_mask"],
            )
            predictions.append(
                norm_stats.denormalize_tensor(predicted).detach().cpu().float().numpy()
            )
            targets.append(norm_stats.denormalize_tensor(teacher).detach().cpu().float().numpy())
            batch_clip_ids = batch.get("clip_id")
            if not isinstance(batch_clip_ids, list):
                raise TypeError("Stage 3 batch clip_id must be a list")
            clip_ids.extend(str(clip_id) for clip_id in cast(list[object], batch_clip_ids))

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination,
        clip_ids=np.asarray(clip_ids),
        predictions=np.concatenate(predictions, axis=0),
        targets=np.concatenate(targets, axis=0),
    )
