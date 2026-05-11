"""Strict loaders and validators for the Stage 1 teacher dump.

The Stage 2+ code treats the teacher dump as read-only ground truth. Validation
therefore fails with clip IDs instead of repairing malformed entries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

TOP_K_LOGITS = 32
WAYPOINTS = 64
TRAJECTORY_DIM = 3
TEACHER_SAMPLES = 16


class ClipValidationError(ValueError):
    """Raised when a teacher dump clip violates the on-disk contract."""


class TeacherDumpExample(TypedDict):
    """Tensor-ready Stage 2 example loaded from one teacher clip."""

    clip_id: str
    frame_paths: list[Path]
    coc_text: str
    token_ids: torch.Tensor
    top_k_token_ids: torch.Tensor
    top_k_logits: torch.Tensor
    teacher_hidden_states: torch.Tensor
    token_mask: torch.Tensor
    hidden_mask: torch.Tensor
    conditioning_meta: dict[str, Any] | None
    kv_cache_files: list[Path]


@dataclass(frozen=True)
class CoCTrace:
    """Chain-of-causation token trace for one clip."""

    token_ids: np.ndarray
    top_k_token_ids: np.ndarray
    top_k_logits: np.ndarray
    texts: tuple[str, ...]


@dataclass(frozen=True)
class ConditioningMeta:
    """Metadata that ties hidden-state slices to Alpamayo expert conditioning."""

    traj_future_start_offset: int
    prefill_seq_len: int
    rope_deltas: tuple[int, ...]
    attention_mask_shape: tuple[int, ...]
    generated_seq_len: int
    n_diffusion_tokens: int
    kv_cache_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class TeacherClipManifest:
    """Validated manifest for one teacher dump clip."""

    clip_id: str
    root: Path
    meta_path: Path
    frame_paths: tuple[Path, ...]
    coc_trace_path: Path
    hidden_states_path: Path
    denoising_traj_path: Path
    trajectories_path: Path
    hidden_shape: tuple[int, int]
    num_tokens: int
    conditioning_meta: ConditioningMeta | None
    kv_cache_files: tuple[Path, ...]


def _clip_error(clip_id: str, message: str) -> ClipValidationError:
    return ClipValidationError(f"Clip {clip_id}: {message}")


def _load_json(path: Path, clip_id: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise _clip_error(clip_id, f"missing {path.name}") from exc
    except json.JSONDecodeError as exc:
        raise _clip_error(clip_id, f"invalid JSON in {path.name}: {exc}") from exc


def _require_file(path: Path, clip_id: str) -> None:
    if not path.is_file():
        raise _clip_error(clip_id, f"missing required file {path.name}")


def _read_split_file(split_file: Path) -> list[str]:
    data = _load_json(split_file, split_file.stem)
    clip_ids_obj: object = data
    if isinstance(data, list):
        pass
    elif isinstance(data, dict):
        data_obj = cast(dict[str, object], data)
        clip_ids_obj = data_obj.get("clip_ids")
    if not isinstance(clip_ids_obj, list):
        raise ClipValidationError(f"Split {split_file}: expected list or object with clip_ids")
    clip_ids = cast(list[object], clip_ids_obj)
    if not all(isinstance(item, str) and item for item in clip_ids):
        raise ClipValidationError(f"Split {split_file}: clip IDs must be non-empty strings")
    return [item for item in clip_ids if isinstance(item, str) and item]


def _validate_meta(meta: Any, clip_dir: Path) -> str:
    fallback_id = clip_dir.name
    if not isinstance(meta, dict):
        raise _clip_error(fallback_id, "meta.json must contain an object")
    meta_obj = cast(dict[str, object], meta)
    clip_id = meta_obj.get("clip_id")
    if not isinstance(clip_id, str) or not clip_id:
        raise _clip_error(fallback_id, "meta.json missing string clip_id")
    num_frames = meta_obj.get("num_frames")
    if not isinstance(num_frames, int) or num_frames <= 0:
        raise _clip_error(clip_id, "meta.json num_frames must be a positive integer")
    if "fps" not in meta_obj:
        raise _clip_error(clip_id, "meta.json missing fps")
    return clip_id


def _validate_frames(clip_dir: Path, clip_id: str) -> tuple[Path, ...]:
    frames_dir = clip_dir / "frames"
    if not frames_dir.is_dir():
        raise _clip_error(clip_id, "missing frames directory")
    frame_paths = tuple(sorted(frames_dir.glob("*.jpg")))
    if not frame_paths:
        raise _clip_error(clip_id, "frames directory contains no .jpg files")
    return frame_paths


def _validate_hidden_states(path: Path, clip_id: str) -> tuple[int, int]:
    _require_file(path, clip_id)
    try:
        hidden_states = np.load(path, mmap_mode="r")
    except ValueError as exc:
        raise _clip_error(clip_id, f"hidden_states.npy is not a valid NumPy array: {exc}") from exc
    if hidden_states.ndim != 2:
        raise _clip_error(clip_id, "hidden_states.npy must have shape (T, D_h)")
    if hidden_states.shape[0] <= 0 or hidden_states.shape[1] <= 0:
        raise _clip_error(clip_id, "hidden_states.npy dimensions must be positive")
    if hidden_states.dtype not in (np.float16, np.float32):
        raise _clip_error(clip_id, "hidden_states.npy must be fp16 or fp32")
    return int(hidden_states.shape[0]), int(hidden_states.shape[1])


def _trace_arrays(trace: Any, clip_id: str) -> CoCTrace:
    if not isinstance(trace, list) or not trace:
        raise _clip_error(clip_id, "coc_trace.json must contain a non-empty list")
    trace_rows = cast(list[object], trace)

    token_rows: list[int] = []
    top_k_id_rows: list[list[int]] = []
    top_k_logit_rows: list[list[float]] = []
    texts: list[str] = []

    for row_idx, row in enumerate(trace_rows):
        if not isinstance(row, dict):
            raise _clip_error(clip_id, f"coc_trace row {row_idx} must be an object")
        row_obj = cast(dict[str, object], row)
        text = row_obj.get("text")
        token_ids = row_obj.get("token_ids")
        top_k_token_ids = row_obj.get("top_k_token_ids")
        top_k_logits = row_obj.get("top_k_logits")
        if not isinstance(text, str):
            raise _clip_error(clip_id, f"coc_trace row {row_idx} missing text")
        if not isinstance(token_ids, list) or not token_ids:
            raise _clip_error(clip_id, f"coc_trace row {row_idx} token_ids must be non-empty")
        if not isinstance(top_k_token_ids, list) or not isinstance(top_k_logits, list):
            raise _clip_error(
                clip_id,
                f"coc_trace row {row_idx} must include top_k_token_ids and top_k_logits",
            )
        token_id_list = cast(list[object], token_ids)
        top_k_token_id_rows = cast(list[object], top_k_token_ids)
        top_k_logit_rows_obj = cast(list[object], top_k_logits)
        if len(top_k_token_id_rows) != len(token_id_list) or len(top_k_logit_rows_obj) != len(
            token_id_list
        ):
            raise _clip_error(clip_id, f"coc_trace row {row_idx} top-k rows must match token_ids")

        for token_idx, token_id in enumerate(token_id_list):
            top_ids = top_k_token_id_rows[token_idx]
            logits = top_k_logit_rows_obj[token_idx]
            if not isinstance(token_id, int):
                raise _clip_error(clip_id, f"token {token_idx} in row {row_idx} is not an int")
            if not isinstance(top_ids, list) or not isinstance(logits, list):
                raise _clip_error(clip_id, f"top-k token {token_idx} in row {row_idx} is invalid")
            top_ids_list = cast(list[object], top_ids)
            logits_list = cast(list[object], logits)
            if len(top_ids_list) != TOP_K_LOGITS or len(logits_list) != TOP_K_LOGITS:
                raise _clip_error(
                    clip_id,
                    f"top-k token {token_idx} in row {row_idx} must have {TOP_K_LOGITS} entries",
                )
            if not all(isinstance(item, int) for item in top_ids_list):
                raise _clip_error(clip_id, f"top_k_token_ids row {row_idx} contains non-int IDs")
            if not all(isinstance(item, (int, float)) for item in logits_list):
                raise _clip_error(
                    clip_id, f"top_k_logits row {row_idx} contains non-numeric logits"
                )
            token_rows.append(token_id)
            top_k_id_rows.append([int(item) for item in top_ids_list if isinstance(item, int)])
            top_k_logit_rows.append(
                [float(item) for item in logits_list if isinstance(item, (int, float))]
            )
        texts.append(text)

    return CoCTrace(
        token_ids=np.asarray(token_rows, dtype=np.int64),
        top_k_token_ids=np.asarray(top_k_id_rows, dtype=np.int64),
        top_k_logits=np.asarray(top_k_logit_rows, dtype=np.float32),
        texts=tuple(texts),
    )


def _validate_denoising(path: Path, clip_id: str) -> None:
    _require_file(path, clip_id)
    try:
        data = np.load(path)
    except ValueError as exc:
        raise _clip_error(clip_id, f"denoising_traj.npz is invalid: {exc}") from exc
    for key in ("x_t", "t", "v_pred"):
        if key not in data:
            raise _clip_error(clip_id, f"denoising_traj.npz missing {key}")
    x_t = data["x_t"]
    v_pred = data["v_pred"]
    times = data["t"]
    if x_t.ndim != 3 or x_t.shape[1:] != (WAYPOINTS, TRAJECTORY_DIM):
        raise _clip_error(clip_id, "denoising x_t must have shape (S, 64, 3)")
    if v_pred.shape != x_t.shape:
        raise _clip_error(clip_id, "denoising v_pred must match x_t shape")
    if times.ndim != 1 or times.shape[0] != x_t.shape[0]:
        raise _clip_error(clip_id, "denoising t must have shape (S,)")


def _validate_trajectories(path: Path, clip_id: str) -> None:
    _require_file(path, clip_id)
    try:
        trajectories = np.load(path, mmap_mode="r")
    except ValueError as exc:
        raise _clip_error(clip_id, f"trajectories.npy is invalid: {exc}") from exc
    if trajectories.shape != (TEACHER_SAMPLES, WAYPOINTS, TRAJECTORY_DIM):
        raise _clip_error(clip_id, "trajectories.npy must have shape (16, 64, 3)")


def _validate_conditioning(
    clip_dir: Path,
    clip_id: str,
    include_kv_cache: bool,
) -> tuple[ConditioningMeta | None, tuple[Path, ...]]:
    conditioning_dir = clip_dir / "conditioning"
    meta_path = conditioning_dir / "conditioning_meta.json"
    if not conditioning_dir.exists():
        if include_kv_cache:
            raise _clip_error(clip_id, "missing conditioning directory")
        return None, ()
    if not meta_path.is_file():
        if include_kv_cache:
            raise _clip_error(clip_id, "missing conditioning_meta.json")
        return None, ()

    data = _load_json(meta_path, clip_id)
    if not isinstance(data, dict):
        raise _clip_error(clip_id, "conditioning_meta.json must contain an object")
    meta = cast(dict[str, object], data)

    required_ints = [
        "traj_future_start_offset",
        "prefill_seq_len",
        "generated_seq_len",
        "n_diffusion_tokens",
    ]
    for key in required_ints:
        value = meta.get(key)
        if not isinstance(value, int) or value < 0:
            raise _clip_error(clip_id, f"conditioning_meta.json {key} must be a non-negative int")
    rope_deltas = meta.get("rope_deltas")
    attention_mask_shape = meta.get("attention_mask_shape")
    rope_deltas_obj = cast(list[object], rope_deltas) if isinstance(rope_deltas, list) else None
    if rope_deltas_obj is None or not all(isinstance(item, int) for item in rope_deltas_obj):
        raise _clip_error(clip_id, "conditioning_meta.json rope_deltas must be a list of ints")
    attention_mask_shape_obj = (
        cast(list[object], attention_mask_shape) if isinstance(attention_mask_shape, list) else None
    )
    if attention_mask_shape_obj is None or not all(
        isinstance(item, int) for item in attention_mask_shape_obj
    ):
        raise _clip_error(
            clip_id,
            "conditioning_meta.json attention_mask_shape must be a list of ints",
        )
    kv_cache_file_names = meta.get("kv_cache_files", [])
    kv_cache_file_names_obj = (
        cast(list[object], kv_cache_file_names) if isinstance(kv_cache_file_names, list) else None
    )
    if kv_cache_file_names_obj is None or not all(
        isinstance(item, str) for item in kv_cache_file_names_obj
    ):
        raise _clip_error(clip_id, "conditioning_meta.json kv_cache_files must be a list of paths")
    rope_delta_list = [item for item in rope_deltas_obj if isinstance(item, int)]
    attention_mask_shape_list = [item for item in attention_mask_shape_obj if isinstance(item, int)]
    kv_cache_name_list = [item for item in kv_cache_file_names_obj if isinstance(item, str)]

    kv_cache_files = tuple(conditioning_dir / name for name in kv_cache_name_list)
    if include_kv_cache:
        if not kv_cache_files:
            raise _clip_error(clip_id, "include_kv_cache=True requires KV cache files")
        for kv_path in kv_cache_files:
            _require_file(kv_path, clip_id)

    return (
        ConditioningMeta(
            traj_future_start_offset=cast(int, meta["traj_future_start_offset"]),
            prefill_seq_len=cast(int, meta["prefill_seq_len"]),
            rope_deltas=tuple(rope_delta_list),
            attention_mask_shape=tuple(attention_mask_shape_list),
            generated_seq_len=cast(int, meta["generated_seq_len"]),
            n_diffusion_tokens=cast(int, meta["n_diffusion_tokens"]),
            kv_cache_files=tuple(kv_cache_name_list),
        ),
        kv_cache_files,
    )


def validate_teacher_clip(
    clip_dir: str | Path, include_kv_cache: bool = False
) -> TeacherClipManifest:
    """Validate one teacher dump clip and return its manifest.

    Args:
        clip_dir: Directory containing a single Stage 1 teacher dump clip.
        include_kv_cache: Require optional KV cache shards under ``conditioning/``.

    Returns:
        A manifest with validated paths and shape metadata.

    Raises:
        ClipValidationError: If any required field is missing or malformed.
    """
    root = Path(clip_dir)
    fallback_id = root.name
    meta_path = root / "meta.json"
    meta = _load_json(meta_path, fallback_id)
    clip_id = _validate_meta(meta, root)
    frame_paths = _validate_frames(root, clip_id)
    coc_trace_path = root / "coc_trace.json"
    trace = _trace_arrays(_load_json(coc_trace_path, clip_id), clip_id)
    hidden_states_path = root / "hidden_states.npy"
    hidden_shape = _validate_hidden_states(hidden_states_path, clip_id)
    denoising_traj_path = root / "denoising_traj.npz"
    trajectories_path = root / "trajectories.npy"
    _validate_denoising(denoising_traj_path, clip_id)
    _validate_trajectories(trajectories_path, clip_id)
    conditioning_meta, kv_cache_files = _validate_conditioning(root, clip_id, include_kv_cache)

    return TeacherClipManifest(
        clip_id=clip_id,
        root=root,
        meta_path=meta_path,
        frame_paths=frame_paths,
        coc_trace_path=coc_trace_path,
        hidden_states_path=hidden_states_path,
        denoising_traj_path=denoising_traj_path,
        trajectories_path=trajectories_path,
        hidden_shape=hidden_shape,
        num_tokens=int(trace.token_ids.shape[0]),
        conditioning_meta=conditioning_meta,
        kv_cache_files=kv_cache_files,
    )


class TeacherDumpDataset(Dataset[TeacherDumpExample]):
    """Dataset that loads Stage 2 fields from the read-only teacher dump."""

    def __init__(
        self,
        root: str | Path,
        split_file: str | Path,
        include_kv_cache: bool = False,
    ) -> None:
        """Initialize a dataset from a dump root and split file.

        Args:
            root: Directory containing one subdirectory per clip ID.
            split_file: JSON list of clip IDs or object with a ``clip_ids`` list.
            include_kv_cache: Require and expose optional teacher KV cache shards.
        """
        self.root = Path(root)
        self.split_file = Path(split_file)
        self.include_kv_cache = include_kv_cache
        self.clip_ids = _read_split_file(self.split_file)
        self.manifests = [
            validate_teacher_clip(self.root / clip_id, include_kv_cache=include_kv_cache)
            for clip_id in self.clip_ids
        ]

    def __len__(self) -> int:
        """Return the number of clips in the split."""
        return len(self.manifests)

    def __getitem__(self, index: int) -> TeacherDumpExample:
        """Load one Stage 2 training example."""
        manifest = self.manifests[index]
        trace = _trace_arrays(
            _load_json(manifest.coc_trace_path, manifest.clip_id), manifest.clip_id
        )
        hidden_states = np.load(manifest.hidden_states_path)
        token_count = int(trace.token_ids.shape[0])
        hidden_count = int(hidden_states.shape[0])

        return {
            "clip_id": manifest.clip_id,
            "frame_paths": list(manifest.frame_paths),
            "coc_text": "\n".join(trace.texts),
            "token_ids": torch.as_tensor(trace.token_ids, dtype=torch.long),
            "top_k_token_ids": torch.as_tensor(trace.top_k_token_ids, dtype=torch.long),
            "top_k_logits": torch.as_tensor(trace.top_k_logits, dtype=torch.float32),
            "teacher_hidden_states": torch.as_tensor(hidden_states, dtype=torch.float32),
            "token_mask": torch.ones(token_count, dtype=torch.bool),
            "hidden_mask": torch.ones(hidden_count, dtype=torch.bool),
            "conditioning_meta": (
                None
                if manifest.conditioning_meta is None
                else {
                    "traj_future_start_offset": manifest.conditioning_meta.traj_future_start_offset,
                    "prefill_seq_len": manifest.conditioning_meta.prefill_seq_len,
                    "rope_deltas": manifest.conditioning_meta.rope_deltas,
                    "attention_mask_shape": manifest.conditioning_meta.attention_mask_shape,
                    "generated_seq_len": manifest.conditioning_meta.generated_seq_len,
                    "n_diffusion_tokens": manifest.conditioning_meta.n_diffusion_tokens,
                    "kv_cache_files": manifest.conditioning_meta.kv_cache_files,
                }
            ),
            "kv_cache_files": list(manifest.kv_cache_files),
        }


def collate_teacher_examples(examples: list[TeacherDumpExample]) -> dict[str, Any]:
    """Pad variable-length Stage 2 examples into a batch."""
    if not examples:
        raise ValueError("Cannot collate an empty example list")

    top_k_widths = {int(example["top_k_token_ids"].shape[1]) for example in examples}
    hidden_dims = {int(example["teacher_hidden_states"].shape[1]) for example in examples}
    if len(top_k_widths) != 1:
        raise ValueError("All examples must have the same top-k width")
    if len(hidden_dims) != 1:
        raise ValueError("All examples must have the same teacher hidden dimension")

    return {
        "clip_id": [example["clip_id"] for example in examples],
        "frame_paths": [example["frame_paths"] for example in examples],
        "coc_text": [example["coc_text"] for example in examples],
        "token_ids": pad_sequence(
            [example["token_ids"] for example in examples],
            batch_first=True,
            padding_value=0,
        ),
        "top_k_token_ids": pad_sequence(
            [example["top_k_token_ids"] for example in examples],
            batch_first=True,
            padding_value=0,
        ),
        "top_k_logits": pad_sequence(
            [example["top_k_logits"] for example in examples],
            batch_first=True,
            padding_value=0.0,
        ),
        "teacher_hidden_states": pad_sequence(
            [example["teacher_hidden_states"] for example in examples],
            batch_first=True,
            padding_value=0.0,
        ),
        "token_mask": pad_sequence(
            [example["token_mask"] for example in examples],
            batch_first=True,
            padding_value=False,
        ),
        "hidden_mask": pad_sequence(
            [example["hidden_mask"] for example in examples],
            batch_first=True,
            padding_value=False,
        ),
        "conditioning_meta": [example["conditioning_meta"] for example in examples],
        "kv_cache_files": [example["kv_cache_files"] for example in examples],
    }
