"""Schema helpers for Alpamayo teacher-conditioning artifacts.

Alpamayo's released inference path exposes the pieces needed by the Stage 1
dump writer: generated token IDs/logits from ``vlm.generate`` plus the
``past_key_values`` cache, RoPE deltas, and ``<|traj_future_start|>`` offset
used immediately before expert diffusion tokens are appended.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

ALPAMAYO_CONDITIONING_META_KEYS = (
    "traj_future_start_offset",
    "prefill_seq_len",
    "rope_deltas",
    "attention_mask_shape",
    "generated_seq_len",
    "n_diffusion_tokens",
    "kv_cache_files",
)


@dataclass(frozen=True)
class AlpamayoConditioningRecord:
    """Serializable record for teacher expert-conditioning metadata."""

    traj_future_start_offset: int
    prefill_seq_len: int
    rope_deltas: tuple[int, ...]
    attention_mask_shape: tuple[int, ...]
    generated_seq_len: int
    n_diffusion_tokens: int
    kv_cache_files: tuple[str, ...]

    def to_json_dict(self) -> dict[str, int | list[int] | list[str]]:
        """Return the JSON representation stored as conditioning_meta.json."""
        return {
            "traj_future_start_offset": self.traj_future_start_offset,
            "prefill_seq_len": self.prefill_seq_len,
            "rope_deltas": list(self.rope_deltas),
            "attention_mask_shape": list(self.attention_mask_shape),
            "generated_seq_len": self.generated_seq_len,
            "n_diffusion_tokens": self.n_diffusion_tokens,
            "kv_cache_files": list(self.kv_cache_files),
        }


@dataclass(frozen=True)
class Stage2ReplayMasks:
    """Boolean replay masks that align student outputs to teacher dump tensors."""

    hidden_position_mask: torch.Tensor
    logit_position_mask: torch.Tensor


def _read_non_negative_int(meta: Mapping[str, Any], key: str) -> int:
    value = meta.get(key)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"conditioning metadata {key} must be a non-negative int")
    return value


def build_stage2_replay_masks(
    *,
    sequence_length: int,
    token_count: int,
    teacher_hidden_length: int,
    conditioning_meta: Mapping[str, Any] | None,
) -> Stage2ReplayMasks:
    """Build masks matching the Stage 1 teacher replay convention.

    Stage 1 replays the full VLM sequence and stores final-layer hidden states
    from the first token through the first ``<|traj_future_start|>`` token. CoC
    logits, however, correspond only to generated CoC tokens. The returned masks
    select those two different views from one student replay.

    Args:
        sequence_length: Student replay token length, shape dimension ``L``.
        token_count: Number of generated CoC tokens in ``coc_trace.json``.
        teacher_hidden_length: First dimension of ``hidden_states.npy``.
        conditioning_meta: Optional ``conditioning_meta.json`` contents.

    Returns:
        Hidden-state and logit masks, each shape ``(L,)``.

    Raises:
        ValueError: If metadata cannot be represented by the replay sequence.
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    if teacher_hidden_length <= 0:
        raise ValueError("teacher_hidden_length must be positive")

    hidden_mask = torch.zeros(sequence_length, dtype=torch.bool)
    logit_mask = torch.zeros(sequence_length, dtype=torch.bool)

    if conditioning_meta is None:
        if teacher_hidden_length != sequence_length:
            raise ValueError(
                "teacher_hidden_length must equal replay length when conditioning metadata "
                "is unavailable"
            )
        if token_count != sequence_length:
            raise ValueError(
                "token_count must equal replay length when conditioning metadata is unavailable"
            )
        hidden_mask[:] = True
        logit_mask[:] = True
        return Stage2ReplayMasks(
            hidden_position_mask=hidden_mask,
            logit_position_mask=logit_mask,
        )

    offset = _read_non_negative_int(conditioning_meta, "traj_future_start_offset")
    prefill_seq_len = _read_non_negative_int(conditioning_meta, "prefill_seq_len")
    if offset != teacher_hidden_length:
        raise ValueError(
            "teacher hidden length must match traj_future_start_offset; "
            f"got {teacher_hidden_length} and {offset}"
        )
    if offset > sequence_length:
        raise ValueError(
            "replay sequence is shorter than the teacher conditioning offset; "
            f"got sequence length {sequence_length} and offset {offset}"
        )

    logit_start = max(prefill_seq_len - 1, 0)
    logit_stop = logit_start + token_count
    if logit_stop > sequence_length:
        raise ValueError(
            "replay sequence is too short for generated-token logit positions; "
            f"need position {logit_stop - 1}, length is {sequence_length}"
        )

    hidden_mask[:offset] = True
    logit_mask[logit_start:logit_stop] = True
    return Stage2ReplayMasks(
        hidden_position_mask=hidden_mask,
        logit_position_mask=logit_mask,
    )
