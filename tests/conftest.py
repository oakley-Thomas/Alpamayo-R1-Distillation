"""Shared test fixtures for synthetic teacher dumps."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from src.data.teacher_dump import TOP_K_LOGITS


Corruption = Literal["missing_hidden", "malformed_topk", "missing_kv_file"]
MakeDump = Callable[..., tuple[Path, Path]]


def _top_k_rows(token_ids: list[int]) -> tuple[list[list[int]], list[list[float]]]:
    all_ids: list[list[int]] = []
    all_logits: list[list[float]] = []
    for token_id in token_ids:
        row_ids = list(range(TOP_K_LOGITS))
        row_ids[0] = token_id
        row_logits = [-20.0 for _ in range(TOP_K_LOGITS)]
        row_logits[0] = 20.0
        all_ids.append(row_ids)
        all_logits.append(row_logits)
    return all_ids, all_logits


def _write_clip(
    dump_root: Path,
    clip_id: str,
    corruption: Corruption | None = None,
    include_kv: bool = False,
) -> None:
    clip_dir = dump_root / clip_id
    frames_dir = clip_dir / "frames"
    frames_dir.mkdir(parents=True)
    (frames_dir / "frame_0000_front.jpg").write_bytes(b"synthetic-jpeg")

    (clip_dir / "meta.json").write_text(
        json.dumps(
            {
                "clip_id": clip_id,
                "num_frames": 1,
                "fps": 10,
                "camera_intrinsics": [[1.0, 0.0, 0.0]],
                "source_url": "fixture",
            }
        ),
        encoding="utf-8",
    )

    token_ids = [3, 4, 5]
    top_k_token_ids, top_k_logits = _top_k_rows(token_ids)
    if corruption == "malformed_topk":
        top_k_logits[0] = top_k_logits[0][:-1]
    (clip_dir / "coc_trace.json").write_text(
        json.dumps(
            [
                {
                    "step_idx": 0,
                    "text": "yield to pedestrian",
                    "token_ids": token_ids,
                    "top_k_token_ids": top_k_token_ids,
                    "top_k_logits": top_k_logits,
                }
            ]
        ),
        encoding="utf-8",
    )

    hidden_states = np.arange(12, dtype=np.float16).reshape(3, 4)
    if corruption != "missing_hidden":
        np.save(clip_dir / "hidden_states.npy", hidden_states)
    np.savez(
        clip_dir / "denoising_traj.npz",
        x_t=np.zeros((2, 64, 3), dtype=np.float32),
        t=np.asarray([0.0, 1.0], dtype=np.float32),
        v_pred=np.ones((2, 64, 3), dtype=np.float32),
    )
    np.save(clip_dir / "trajectories.npy", np.zeros((16, 64, 3), dtype=np.float32))

    if include_kv or corruption == "missing_kv_file":
        conditioning_dir = clip_dir / "conditioning"
        conditioning_dir.mkdir()
        kv_files = ["kv_layer_000.npz"]
        if include_kv and corruption != "missing_kv_file":
            np.savez(
                conditioning_dir / kv_files[0],
                key=np.zeros((1, 2, 3), dtype=np.float16),
                value=np.zeros((1, 2, 3), dtype=np.float16),
            )
        (conditioning_dir / "conditioning_meta.json").write_text(
            json.dumps(
                {
                    "traj_future_start_offset": 8,
                    "prefill_seq_len": 16,
                    "rope_deltas": [0],
                    "attention_mask_shape": [1, 1, 64, 80],
                    "generated_seq_len": 3,
                    "n_diffusion_tokens": 64,
                    "kv_cache_files": kv_files,
                }
            ),
            encoding="utf-8",
        )


@pytest.fixture
def mini_dump(tmp_path: Path) -> tuple[Path, Path]:
    """Create a valid two-clip teacher dump and split file."""
    dump_root = tmp_path / "teacher_dump"
    dump_root.mkdir()
    clip_ids = ["clip-a", "clip-b"]
    for clip_id in clip_ids:
        _write_clip(dump_root, clip_id)
    split_file = tmp_path / "train.json"
    split_file.write_text(json.dumps(clip_ids), encoding="utf-8")
    return dump_root, split_file


@pytest.fixture
def make_dump(tmp_path: Path) -> MakeDump:
    """Return a helper that builds a one-clip dump with optional corruption."""

    def _make(
        corruption: Corruption | None = None,
        include_kv: bool = False,
    ) -> tuple[Path, Path]:
        dump_root = tmp_path / f"teacher_dump_{corruption or 'valid'}_{include_kv}"
        dump_root.mkdir()
        _write_clip(dump_root, "clip-a", corruption=corruption, include_kv=include_kv)
        split_file = tmp_path / f"split_{corruption or 'valid'}_{include_kv}.json"
        split_file.write_text(json.dumps(["clip-a"]), encoding="utf-8")
        return dump_root, split_file

    return _make
