"""Stage 3 dataset and trajectory-normalization utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.data.teacher_dump import (
    TEACHER_SAMPLES,
    TRAJECTORY_DIM,
    WAYPOINTS,
    ClipValidationError,
    TeacherClipManifest,
    TeacherDumpDataset,
)

TRAJECTORY_STD_EPSILON = 1e-6
ConditioningSource = Literal["student_cache", "teacher_dump"]
STUDENT_CACHE_CONDITIONING: ConditioningSource = "student_cache"
TEACHER_DUMP_CONDITIONING: ConditioningSource = "teacher_dump"


class Stage3Example(TypedDict):
    """Tensor-ready Stage 3 example loaded from one clip."""

    clip_id: str
    teacher_trajectories: torch.Tensor
    conditioning_hidden_states: torch.Tensor
    hidden_mask: torch.Tensor


@dataclass(frozen=True)
class TrajectoryNormStats:
    """Per-axis trajectory normalization statistics for Stage 3."""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    def to_json_dict(self) -> dict[str, list[float]]:
        """Return a JSON-serializable representation."""
        return {"mean": list(self.mean), "std": list(self.std)}

    @classmethod
    def from_json_dict(cls, data: Any) -> TrajectoryNormStats:
        """Build stats from a JSON-loaded object."""
        if not isinstance(data, dict):
            raise ValueError("trajectory norm stats must be a JSON object")
        data_obj = cast(dict[str, object], data)
        mean = _read_float_triplet(data_obj.get("mean"), "mean")
        std = _read_float_triplet(data_obj.get("std"), "std")
        if any(value <= 0.0 for value in std):
            raise ValueError("trajectory norm std values must be positive")
        return cls(mean=mean, std=std)

    def save(self, path: str | Path) -> None:
        """Write stats to disk as JSON."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_json_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> TrajectoryNormStats:
        """Load stats from a JSON file."""
        source = Path(path)
        with source.open("r", encoding="utf-8") as handle:
            return cls.from_json_dict(json.load(handle))

    def normalize_tensor(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Normalize trajectory tensors with shape (..., 3)."""
        mean, std = self._torch_stats(trajectories)
        return (trajectories - mean) / std

    def denormalize_tensor(self, trajectories: torch.Tensor) -> torch.Tensor:
        """De-normalize trajectory tensors with shape (..., 3)."""
        mean, std = self._torch_stats(trajectories)
        return trajectories * std + mean

    def normalize_array(self, trajectories: np.ndarray) -> np.ndarray:
        """Normalize trajectory arrays with shape (..., 3)."""
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        return ((trajectories.astype(np.float32, copy=False) - mean) / std).astype(np.float32)

    def _torch_stats(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = reference.new_tensor(self.mean)
        std = reference.new_tensor(self.std)
        return mean, std


class Stage3TrajectoryDataset(Dataset[Stage3Example]):
    """Dataset that pairs teacher trajectories with frozen conditioning states."""

    def __init__(
        self,
        teacher_dump_root: str | Path,
        split_file: str | Path,
        hidden_cache_dir: str | Path | None = None,
        norm_stats: TrajectoryNormStats | None = None,
        conditioning_source: str = STUDENT_CACHE_CONDITIONING,
    ) -> None:
        """Initialize a Stage 3 dataset.

        Args:
            teacher_dump_root: Directory containing one Stage 1 dump directory per clip.
            split_file: JSON list of clip IDs or object with a ``clip_ids`` list.
            hidden_cache_dir: Directory containing ``<clip_id>.npy`` Stage 2 caches.
            norm_stats: Optional global train stats used to normalize trajectories.
            conditioning_source: ``student_cache`` reads Stage 2 cache files;
                ``teacher_dump`` reads ``hidden_states.npy`` directly from each
                teacher dump clip.
        """
        self.teacher_dataset = TeacherDumpDataset(teacher_dump_root, split_file)
        self.conditioning_source = _validate_conditioning_source(conditioning_source)
        self.hidden_cache_dir = None if hidden_cache_dir is None else Path(hidden_cache_dir)
        self.norm_stats = norm_stats
        self.manifests = self.teacher_dataset.manifests
        self.hidden_cache_paths: tuple[Path, ...] = ()
        if self.conditioning_source == STUDENT_CACHE_CONDITIONING:
            if self.hidden_cache_dir is None:
                raise ValueError("hidden_cache_dir is required for student_cache conditioning")
            self.hidden_cache_paths = tuple(
                _validate_hidden_cache(self.hidden_cache_dir, manifest)
                for manifest in self.manifests
            )
            self.hidden_dim = _infer_cache_hidden_dim(self.hidden_cache_paths)
        else:
            self.hidden_dim = _infer_teacher_hidden_dim(tuple(self.manifests))

    def __len__(self) -> int:
        """Return the number of clips in the split."""
        return len(self.manifests)

    def __getitem__(self, index: int) -> Stage3Example:
        """Load one clip worth of Stage 3 training data."""
        manifest = self.manifests[index]
        trajectories = _load_trajectories(manifest)
        if self.norm_stats is not None:
            trajectories = self.norm_stats.normalize_array(trajectories)
        hidden_states = self._load_conditioning_hidden_states(index, manifest)
        hidden_count = int(hidden_states.shape[0])
        return {
            "clip_id": manifest.clip_id,
            "teacher_trajectories": torch.as_tensor(trajectories, dtype=torch.float32),
            "conditioning_hidden_states": torch.as_tensor(hidden_states, dtype=torch.float32),
            "hidden_mask": torch.ones(hidden_count, dtype=torch.bool),
        }

    def _load_conditioning_hidden_states(
        self, index: int, manifest: TeacherClipManifest
    ) -> np.ndarray:
        if self.conditioning_source == TEACHER_DUMP_CONDITIONING:
            return _load_teacher_hidden_states(manifest)
        return _load_hidden_cache(self.hidden_cache_paths[index], manifest.clip_id)


def compute_trajectory_norm_stats(dataset: Stage3TrajectoryDataset) -> TrajectoryNormStats:
    """Compute global per-axis trajectory stats from a Stage 3 training dataset."""
    if len(dataset.manifests) == 0:
        raise ValueError("Cannot compute trajectory stats from an empty dataset")

    axis_sum = np.zeros((TRAJECTORY_DIM,), dtype=np.float64)
    axis_sq_sum = np.zeros((TRAJECTORY_DIM,), dtype=np.float64)
    count = 0
    for manifest in dataset.manifests:
        trajectories = _load_trajectories(manifest).astype(np.float64, copy=False)
        flat = trajectories.reshape(-1, TRAJECTORY_DIM)
        axis_sum += flat.sum(axis=0)
        axis_sq_sum += np.square(flat).sum(axis=0)
        count += int(flat.shape[0])

    if count == 0:
        raise ValueError("Cannot compute trajectory stats from zero trajectory values")
    mean = axis_sum / count
    variance = np.maximum(axis_sq_sum / count - np.square(mean), TRAJECTORY_STD_EPSILON)
    std = np.sqrt(variance)
    return TrajectoryNormStats(
        mean=(float(mean[0]), float(mean[1]), float(mean[2])),
        std=(float(std[0]), float(std[1]), float(std[2])),
    )


def collate_stage3_examples(examples: list[Stage3Example]) -> dict[str, Any]:
    """Collate clips into one row per teacher trajectory sample."""
    if not examples:
        raise ValueError("Cannot collate an empty Stage 3 batch")

    hidden_dims = {int(example["conditioning_hidden_states"].shape[1]) for example in examples}
    if len(hidden_dims) != 1:
        raise ValueError("All Stage 3 examples must have the same hidden dimension")

    trajectory_batches: list[torch.Tensor] = []
    hidden_batches: list[torch.Tensor] = []
    mask_batches: list[torch.Tensor] = []
    clip_ids: list[str] = []
    for example in examples:
        trajectories = example["teacher_trajectories"]
        hidden_states = example["conditioning_hidden_states"]
        hidden_mask = example["hidden_mask"]
        _validate_example_tensors(trajectories, hidden_states, hidden_mask, example["clip_id"])
        sample_count = int(trajectories.shape[0])
        trajectory_batches.append(trajectories)
        hidden_batches.append(hidden_states)
        mask_batches.append(hidden_mask)
        clip_ids.extend([example["clip_id"]] * sample_count)

    padded_hidden = pad_sequence(hidden_batches, batch_first=True, padding_value=0.0)
    padded_masks = pad_sequence(mask_batches, batch_first=True, padding_value=False)
    repeated_hidden: list[torch.Tensor] = []
    repeated_masks: list[torch.Tensor] = []
    for batch_idx, trajectories in enumerate(trajectory_batches):
        sample_count = int(trajectories.shape[0])
        repeated_hidden.append(padded_hidden[batch_idx].unsqueeze(0).expand(sample_count, -1, -1))
        repeated_masks.append(padded_masks[batch_idx].unsqueeze(0).expand(sample_count, -1))

    return {
        "clip_id": clip_ids,
        "teacher_trajectories": torch.cat(trajectory_batches, dim=0),
        "conditioning_hidden_states": torch.cat(repeated_hidden, dim=0),
        "hidden_mask": torch.cat(repeated_masks, dim=0),
    }


def _read_float_triplet(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list):
        raise ValueError(f"trajectory norm {name} must contain three values")
    values = cast(list[object], value)
    if len(values) != TRAJECTORY_DIM:
        raise ValueError(f"trajectory norm {name} must contain three values")
    numeric_values: list[float] = []
    for item in values:
        if not isinstance(item, (int, float)):
            raise ValueError(f"trajectory norm {name} values must be numeric")
        numeric_values.append(float(item))
    return (numeric_values[0], numeric_values[1], numeric_values[2])


def _clip_error(clip_id: str, message: str) -> ClipValidationError:
    return ClipValidationError(f"Clip {clip_id}: {message}")


def _validate_conditioning_source(source: str) -> ConditioningSource:
    if source == STUDENT_CACHE_CONDITIONING:
        return STUDENT_CACHE_CONDITIONING
    if source == TEACHER_DUMP_CONDITIONING:
        return TEACHER_DUMP_CONDITIONING
    raise ValueError(
        "Stage 3 conditioning_source must be "
        f"{STUDENT_CACHE_CONDITIONING!r} or {TEACHER_DUMP_CONDITIONING!r}"
    )


def _validate_hidden_cache(hidden_cache_dir: Path, manifest: TeacherClipManifest) -> Path:
    cache_path = hidden_cache_dir / f"{manifest.clip_id}.npy"
    if not cache_path.is_file():
        raise _clip_error(manifest.clip_id, f"missing student hidden cache {cache_path.name}")
    try:
        hidden_states = np.load(cache_path, mmap_mode="r")
    except ValueError as exc:
        raise _clip_error(manifest.clip_id, f"student hidden cache is invalid: {exc}") from exc
    if hidden_states.ndim != 2:
        raise _clip_error(manifest.clip_id, "student hidden cache must have shape (T, D_h)")
    if tuple(int(dim) for dim in hidden_states.shape) != manifest.hidden_shape:
        raise _clip_error(
            manifest.clip_id,
            "student hidden cache shape must match teacher hidden shape; "
            f"got {tuple(int(dim) for dim in hidden_states.shape)} and {manifest.hidden_shape}",
        )
    if hidden_states.dtype not in (np.float16, np.float32):
        raise _clip_error(manifest.clip_id, "student hidden cache must be fp16 or fp32")
    return cache_path


def _infer_cache_hidden_dim(cache_paths: tuple[Path, ...]) -> int:
    if not cache_paths:
        raise ValueError("Stage 3 split is empty")
    hidden_states = np.load(cache_paths[0], mmap_mode="r")
    return int(hidden_states.shape[1])


def _infer_teacher_hidden_dim(manifests: tuple[TeacherClipManifest, ...]) -> int:
    if not manifests:
        raise ValueError("Stage 3 split is empty")
    return manifests[0].hidden_shape[1]


def _load_trajectories(manifest: TeacherClipManifest) -> np.ndarray:
    try:
        trajectories = np.load(manifest.trajectories_path, mmap_mode="r")
    except ValueError as exc:
        raise _clip_error(manifest.clip_id, f"trajectories.npy is invalid: {exc}") from exc
    if trajectories.shape != (TEACHER_SAMPLES, WAYPOINTS, TRAJECTORY_DIM):
        raise _clip_error(manifest.clip_id, "trajectories.npy must have shape (16, 64, 3)")
    return np.array(trajectories, dtype=np.float32, copy=True)


def _load_hidden_cache(cache_path: Path, clip_id: str) -> np.ndarray:
    try:
        hidden_states = np.load(cache_path, mmap_mode="r")
    except ValueError as exc:
        raise _clip_error(clip_id, f"student hidden cache is invalid: {exc}") from exc
    return np.array(hidden_states, dtype=np.float32, copy=True)


def _load_teacher_hidden_states(manifest: TeacherClipManifest) -> np.ndarray:
    try:
        hidden_states = np.load(manifest.hidden_states_path, mmap_mode="r")
    except ValueError as exc:
        raise _clip_error(manifest.clip_id, f"hidden_states.npy is invalid: {exc}") from exc
    if tuple(int(dim) for dim in hidden_states.shape) != manifest.hidden_shape:
        raise _clip_error(
            manifest.clip_id,
            "hidden_states.npy shape changed after validation; "
            f"got {tuple(int(dim) for dim in hidden_states.shape)} and {manifest.hidden_shape}",
        )
    return np.array(hidden_states, dtype=np.float32, copy=True)


def _validate_example_tensors(
    trajectories: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_mask: torch.Tensor,
    clip_id: str,
) -> None:
    if trajectories.shape != (TEACHER_SAMPLES, WAYPOINTS, TRAJECTORY_DIM):
        raise ValueError(f"Clip {clip_id}: teacher trajectories must have shape (16, 64, 3)")
    if hidden_states.ndim != 2:
        raise ValueError(f"Clip {clip_id}: hidden states must have shape (T, D_h)")
    if hidden_mask.shape != hidden_states.shape[:1]:
        raise ValueError(f"Clip {clip_id}: hidden mask must have shape (T,)")
