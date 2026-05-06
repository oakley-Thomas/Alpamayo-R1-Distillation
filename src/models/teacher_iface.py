"""Schema helpers for Alpamayo teacher-conditioning artifacts.

Alpamayo's released inference path exposes the pieces needed by the Stage 1
dump writer: generated token IDs/logits from ``vlm.generate`` plus the
``past_key_values`` cache, RoPE deltas, and ``<|traj_future_start|>`` offset
used immediately before expert diffusion tokens are appended.
"""

from __future__ import annotations

from dataclasses import dataclass


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
