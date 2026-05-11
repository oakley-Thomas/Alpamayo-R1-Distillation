"""Stage 3 trajectory metrics and evaluation runner."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.stage3 import (
    Stage3TrajectoryDataset,
    TrajectoryNormStats,
    collate_stage3_examples,
)
from src.models.action_expert import FlowMatchingActionExpert
from src.train.config import load_stage3_config
from src.train.stage3 import build_stage3_model, load_stage3_checkpoint, resolve_stage3_device

ADE_TARGET_METERS = 0.5
FDE_TARGET_METERS = 1.5
HEADING_CONTINUITY_TARGET = 0.99
LATENCY_TARGET_MS = 50.0
HEADING_JUMP_THRESHOLD_RADIANS = math.pi / 4.0
LATENCY_WARMUP_TRIALS = 10
LATENCY_MEASURE_TRIALS = 100
VRAM_MEASURE_TRIALS = 100
TRAJECTORY_HORIZON_SECONDS = 6.0


@dataclass(frozen=True)
class Stage3PredictionArrays:
    """Predicted and teacher trajectory arrays for Stage 3 eval."""

    clip_ids: tuple[str, ...]
    predictions: np.ndarray
    targets: np.ndarray


@dataclass(frozen=True)
class Stage3EvalReport:
    """Acceptance metrics for Stage 3 Action Expert distillation."""

    split_file: str
    num_trajectories: int
    ade_m: float
    fde_m: float
    heading_continuity_rate: float
    latency_ms: float | None
    passes_ade: bool
    passes_fde: bool
    passes_heading_continuity: bool
    passes_latency: bool | None
    ade_1s_m: float | None = None
    ade_3s_m: float | None = None
    ade_6s_m: float | None = None
    fde_6s_m: float | None = None
    vram_gb: float | None = None

    @property
    def passes_acceptance(self) -> bool:
        """Return whether measured Stage 3 acceptance metrics pass."""
        latency_passed = self.passes_latency is not False
        return (
            self.passes_ade
            and self.passes_fde
            and self.passes_heading_continuity
            and latency_passed
        )

    def to_json_dict(self) -> dict[str, bool | float | int | str | None]:
        """Return a JSON-serializable report dictionary."""
        data = asdict(self)
        data["passes_acceptance"] = self.passes_acceptance
        return cast(dict[str, bool | float | int | str | None], data)


def trajectory_ade(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute average displacement error over x/y waypoint positions.

    Args:
        predictions: Predicted trajectories, shape (B, 64, 3).
        targets: Teacher trajectories, shape (B, 64, 3).

    Returns:
        Mean L2 displacement error over batch and waypoints.
    """
    _validate_trajectory_pair(predictions, targets)
    displacement = predictions[..., :2] - targets[..., :2]
    return torch.sqrt(torch.sum(displacement.square(), dim=-1)).mean()


def trajectory_ade_at_horizon(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    horizon_seconds: float,
) -> torch.Tensor:
    """Compute ADE over waypoints up to a time horizon.

    Args:
        predictions: Predicted trajectories, shape (B, 64, 3).
        targets: Teacher trajectories, shape (B, 64, 3).
        horizon_seconds: Horizon within the 6s, 64-waypoint trajectory.

    Returns:
        Mean L2 displacement error through the requested horizon.
    """
    _validate_trajectory_pair(predictions, targets)
    waypoint_count = _waypoint_count_for_horizon(
        waypoint_count=int(predictions.shape[1]),
        horizon_seconds=horizon_seconds,
    )
    return trajectory_ade(predictions[:, :waypoint_count], targets[:, :waypoint_count])


def trajectory_fde(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute final displacement error over x/y final waypoint positions.

    Args:
        predictions: Predicted trajectories, shape (B, 64, 3).
        targets: Teacher trajectories, shape (B, 64, 3).

    Returns:
        Mean L2 final waypoint displacement error over batch.
    """
    _validate_trajectory_pair(predictions, targets)
    displacement = predictions[:, -1, :2] - targets[:, -1, :2]
    return torch.sqrt(torch.sum(displacement.square(), dim=-1)).mean()


def trajectory_fde_at_horizon(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    horizon_seconds: float,
) -> torch.Tensor:
    """Compute FDE at a requested trajectory horizon.

    Args:
        predictions: Predicted trajectories, shape (B, 64, 3).
        targets: Teacher trajectories, shape (B, 64, 3).
        horizon_seconds: Horizon within the 6s, 64-waypoint trajectory.

    Returns:
        Mean L2 displacement error at the final waypoint for the horizon.
    """
    _validate_trajectory_pair(predictions, targets)
    waypoint_count = _waypoint_count_for_horizon(
        waypoint_count=int(predictions.shape[1]),
        horizon_seconds=horizon_seconds,
    )
    return trajectory_fde(predictions[:, :waypoint_count], targets[:, :waypoint_count])


def heading_continuity_rate(
    predictions: torch.Tensor,
    threshold_radians: float = HEADING_JUMP_THRESHOLD_RADIANS,
) -> torch.Tensor:
    """Compute the rate of trajectories without large adjacent heading jumps.

    Args:
        predictions: Predicted trajectories, shape (B, 64, 3).
        threshold_radians: Maximum allowed wrapped heading delta.

    Returns:
        Fraction of trajectories with all adjacent heading changes within threshold.
    """
    if predictions.ndim != 3 or predictions.shape[-1] != 3:
        raise ValueError("predictions must have shape (B, 64, 3)")
    if threshold_radians <= 0.0:
        raise ValueError("heading threshold must be positive")
    heading = predictions[..., 2]
    deltas = heading[:, 1:] - heading[:, :-1]
    wrapped = torch.atan2(torch.sin(deltas), torch.cos(deltas)).abs()
    continuous = (wrapped <= threshold_radians).all(dim=1)
    return continuous.to(dtype=predictions.dtype).mean()


def evaluate_stage3_predictions(
    *,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    split_file: str,
    latency_ms: float | None = None,
    vram_gb: float | None = None,
) -> Stage3EvalReport:
    """Compute Stage 3 acceptance metrics from prediction tensors."""
    _validate_trajectory_pair(predictions, targets)
    ade_1s = float(trajectory_ade_at_horizon(predictions, targets, 1.0).item())
    ade_3s = float(trajectory_ade_at_horizon(predictions, targets, 3.0).item())
    ade_6s = float(trajectory_ade_at_horizon(predictions, targets, 6.0).item())
    fde_6s = float(trajectory_fde_at_horizon(predictions, targets, 6.0).item())
    heading_rate = float(heading_continuity_rate(predictions).item())
    return Stage3EvalReport(
        split_file=split_file,
        num_trajectories=int(predictions.shape[0]),
        ade_m=ade_6s,
        fde_m=fde_6s,
        heading_continuity_rate=heading_rate,
        latency_ms=latency_ms,
        passes_ade=ade_6s <= ADE_TARGET_METERS,
        passes_fde=fde_6s <= FDE_TARGET_METERS,
        passes_heading_continuity=heading_rate >= HEADING_CONTINUITY_TARGET,
        passes_latency=None if latency_ms is None else latency_ms <= LATENCY_TARGET_MS,
        ade_1s_m=ade_1s,
        ade_3s_m=ade_3s,
        ade_6s_m=ade_6s,
        fde_6s_m=fde_6s,
        vram_gb=vram_gb,
    )


def generate_stage3_predictions(
    *,
    model: FlowMatchingActionExpert,
    dataset: Stage3TrajectoryDataset,
    norm_stats: TrajectoryNormStats,
    device: torch.device,
    seed: int,
) -> Stage3PredictionArrays:
    """Run one-step Stage 3 inference and return de-normalized trajectories."""
    if len(dataset) == 0:
        raise ValueError("Cannot evaluate an empty Stage 3 dataset")
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_stage3_examples)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    clip_ids: list[str] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            tensor_batch = _move_eval_batch(batch, device)
            teacher = tensor_batch["teacher_trajectories"]
            noise = torch.randn(
                teacher.shape,
                device=teacher.device,
                dtype=teacher.dtype,
                generator=generator,
            )
            predicted = model.single_step(
                noise,
                tensor_batch["conditioning_hidden_states"],
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

    return Stage3PredictionArrays(
        clip_ids=tuple(clip_ids),
        predictions=np.concatenate(predictions, axis=0),
        targets=np.concatenate(targets, axis=0),
    )


def measure_stage3_latency_ms(
    *,
    model: FlowMatchingActionExpert,
    dataset: Stage3TrajectoryDataset,
    device: torch.device,
    warmup_trials: int = LATENCY_WARMUP_TRIALS,
    measure_trials: int = LATENCY_MEASURE_TRIALS,
) -> float:
    """Measure single-step Action Expert latency in milliseconds."""
    if len(dataset) == 0:
        raise ValueError("Cannot measure latency on an empty Stage 3 dataset")
    if warmup_trials < 0 or measure_trials <= 0:
        raise ValueError("Latency trial counts are invalid")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_stage3_examples)
    first_batch = next(iter(loader))
    tensor_batch = _move_eval_batch(first_batch, device)
    teacher = tensor_batch["teacher_trajectories"][:1]
    hidden_states = tensor_batch["conditioning_hidden_states"][:1]
    hidden_mask = tensor_batch["hidden_mask"][:1]
    noise = torch.randn(teacher.shape, device=device, dtype=teacher.dtype)
    model.eval()

    with torch.no_grad():
        for _ in range(warmup_trials):
            model.single_step(noise, hidden_states, hidden_mask)
        _synchronize_if_cuda(device)
        start = time.perf_counter()
        for _ in range(measure_trials):
            model.single_step(noise, hidden_states, hidden_mask)
        _synchronize_if_cuda(device)
        elapsed_seconds = time.perf_counter() - start
    return elapsed_seconds * 1000.0 / measure_trials


def measure_stage3_vram_gb(
    *,
    model: FlowMatchingActionExpert,
    dataset: Stage3TrajectoryDataset,
    device: torch.device,
    measure_trials: int = VRAM_MEASURE_TRIALS,
) -> float:
    """Measure peak CUDA memory allocated during single-step Action Expert inference."""
    if device.type != "cuda":
        raise RuntimeError("Stage 3 VRAM measurement requires a CUDA device")
    if len(dataset) == 0:
        raise ValueError("Cannot measure VRAM on an empty Stage 3 dataset")
    if measure_trials <= 0:
        raise ValueError("VRAM trial count must be positive")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_stage3_examples)
    first_batch = next(iter(loader))
    tensor_batch = _move_eval_batch(first_batch, device)
    teacher = tensor_batch["teacher_trajectories"][:1]
    hidden_states = tensor_batch["conditioning_hidden_states"][:1]
    hidden_mask = tensor_batch["hidden_mask"][:1]
    noise = torch.randn(teacher.shape, device=device, dtype=teacher.dtype)
    model.eval()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(measure_trials):
            model.single_step(noise, hidden_states, hidden_mask)
        _synchronize_if_cuda(device)
    bytes_per_gibibyte = 1024.0**3
    return float(torch.cuda.max_memory_allocated(device) / bytes_per_gibibyte)


def write_stage3_predictions(predictions: Stage3PredictionArrays, output_path: str | Path) -> None:
    """Write Stage 3 predictions and teacher targets to an NPZ file."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination,
        clip_ids=np.asarray(predictions.clip_ids),
        predictions=predictions.predictions,
        targets=predictions.targets,
    )


def write_stage3_eval_report(report: Stage3EvalReport, output_path: str | Path) -> None:
    """Write a Stage 3 eval report as JSON."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.to_json_dict(), indent=2) + "\n",
        encoding="utf-8",
    )


def run_stage3_evaluation(
    *,
    config_path: str | Path,
    split_file: str | None = None,
    checkpoint_path: str | None = None,
    predictions_output: str | None = None,
    measure_latency: bool = False,
    measure_vram: bool = False,
) -> Stage3EvalReport:
    """Run Stage 3 evaluation from a saved checkpoint."""
    config = load_stage3_config(config_path)
    device = resolve_stage3_device(config)
    resolved_split = split_file or config.data.val_split
    resolved_checkpoint = checkpoint_path or config.outputs.checkpoint_path
    raw_dataset = Stage3TrajectoryDataset(
        teacher_dump_root=config.data.teacher_dump_root,
        split_file=resolved_split,
        hidden_cache_dir=config.data.hidden_cache_dir,
        conditioning_source=config.data.conditioning_source,
    )
    model = build_stage3_model(config, teacher_hidden_dim=raw_dataset.hidden_dim).to(device)
    norm_stats = load_stage3_checkpoint(model, resolved_checkpoint, map_location=device)
    dataset = Stage3TrajectoryDataset(
        teacher_dump_root=config.data.teacher_dump_root,
        split_file=resolved_split,
        hidden_cache_dir=config.data.hidden_cache_dir,
        norm_stats=norm_stats,
        conditioning_source=config.data.conditioning_source,
    )
    prediction_arrays = generate_stage3_predictions(
        model=model,
        dataset=dataset,
        norm_stats=norm_stats,
        device=device,
        seed=config.training.seed,
    )
    if predictions_output is not None:
        write_stage3_predictions(prediction_arrays, predictions_output)

    latency_ms = (
        measure_stage3_latency_ms(model=model, dataset=dataset, device=device)
        if measure_latency
        else None
    )
    vram_gb = (
        measure_stage3_vram_gb(model=model, dataset=dataset, device=device)
        if measure_vram
        else None
    )
    return evaluate_stage3_predictions(
        predictions=torch.as_tensor(prediction_arrays.predictions),
        targets=torch.as_tensor(prediction_arrays.targets),
        split_file=resolved_split,
        latency_ms=latency_ms,
        vram_gb=vram_gb,
    )


def _validate_trajectory_pair(predictions: torch.Tensor, targets: torch.Tensor) -> None:
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have matching shapes")
    if predictions.ndim != 3 or predictions.shape[-1] != 3:
        raise ValueError("trajectory tensors must have shape (B, 64, 3)")


def _waypoint_count_for_horizon(*, waypoint_count: int, horizon_seconds: float) -> int:
    if waypoint_count <= 0:
        raise ValueError("waypoint_count must be positive")
    if horizon_seconds <= 0.0:
        raise ValueError("horizon_seconds must be positive")
    scaled_count = math.ceil(waypoint_count * horizon_seconds / TRAJECTORY_HORIZON_SECONDS)
    return min(max(scaled_count, 1), waypoint_count)


def _move_eval_batch(batch: dict[str, object], device: torch.device) -> dict[str, torch.Tensor]:
    tensor_batch: dict[str, torch.Tensor] = {}
    for key in ("teacher_trajectories", "conditioning_hidden_states", "hidden_mask"):
        value = batch.get(key)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Stage 3 batch field {key} must be a tensor")
        tensor_batch[key] = value.to(device)
    return tensor_batch


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
