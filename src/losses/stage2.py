"""Stage 2 VLM distillation losses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class Stage2LossConfig:
    """Configuration for Stage 2 loss weighting."""

    alpha: float = 1.0
    beta: float = 0.1
    temperature: float = 2.0


@dataclass(frozen=True)
class Stage2LossOutput:
    """Named Stage 2 loss components."""

    total: torch.Tensor
    coc_kl: torch.Tensor
    hidden_align: torch.Tensor
    lm_ce: torch.Tensor


@dataclass(frozen=True)
class Stage2ModelOutput:
    """Minimal model output consumed by Stage 2 loss computation."""

    logits: torch.Tensor
    adapted_hidden_states: torch.Tensor


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    denominator = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denominator


def _check_finite(name: str, value: torch.Tensor) -> None:
    isfinite = torch.isfinite
    if not isfinite(value).all():
        raise FloatingPointError(f"{name} became non-finite")


def _coc_top_k_kl(
    student_logits: torch.Tensor,
    top_k_token_ids: torch.Tensor,
    teacher_top_k_logits: torch.Tensor,
    token_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if student_logits.ndim != 3:
        raise ValueError("student logits must have shape (B, L, V)")
    if top_k_token_ids.shape != teacher_top_k_logits.shape:
        raise ValueError("top_k_token_ids and top_k_logits must have matching shapes")
    if top_k_token_ids.ndim != 3:
        raise ValueError("top-k tensors must have shape (B, L, K)")
    if student_logits.shape[:2] != top_k_token_ids.shape[:2]:
        raise ValueError("student logits and top-k tensors must share batch/token dimensions")

    student_selected = student_logits.gather(dim=-1, index=top_k_token_ids)
    teacher_log_probs = F.log_softmax(teacher_top_k_logits / temperature, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    student_log_probs = F.log_softmax(student_selected / temperature, dim=-1)
    per_token_kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    return _masked_mean(per_token_kl, token_mask)


def _hidden_alignment_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    hidden_mask: torch.Tensor,
) -> torch.Tensor:
    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError(
            "student and teacher hidden states must have matching shape; "
            f"got {tuple(student_hidden.shape)} and {tuple(teacher_hidden.shape)}"
        )
    per_element = F.smooth_l1_loss(student_hidden, teacher_hidden, reduction="none")
    return _masked_mean(per_element, hidden_mask)


def _lm_ce_loss(
    student_logits: torch.Tensor,
    token_ids: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    if student_logits.shape[:2] != token_ids.shape:
        raise ValueError("student logits and token_ids must share batch/token dimensions")
    vocab_size = student_logits.shape[-1]
    flat_loss = F.cross_entropy(
        student_logits.reshape(-1, vocab_size),
        token_ids.reshape(-1),
        reduction="none",
    )
    return _masked_mean(flat_loss.reshape_as(token_ids), token_mask)


def compute_stage2_loss(
    batch: Mapping[str, torch.Tensor],
    outputs: Stage2ModelOutput,
    config: Stage2LossConfig,
) -> Stage2LossOutput:
    """Compute the Stage 2 distillation objective.

    Args:
        batch: Tensor batch containing token IDs, top-k teacher logits/IDs,
            teacher hidden states, and masks.
        outputs: Student logits of shape (B, L, V) and adapted hidden states of
            shape (B, T, D_h).
        config: Loss weights and temperature.

    Returns:
        Total loss and the named CoC-KL, hidden alignment, and LM-CE components.

    Raises:
        FloatingPointError: If any component becomes non-finite.
    """
    coc_kl = _coc_top_k_kl(
        student_logits=outputs.logits,
        top_k_token_ids=batch["top_k_token_ids"],
        teacher_top_k_logits=batch["top_k_logits"],
        token_mask=batch["token_mask"],
        temperature=config.temperature,
    )
    hidden_align = _hidden_alignment_loss(
        student_hidden=outputs.adapted_hidden_states,
        teacher_hidden=batch["teacher_hidden_states"],
        hidden_mask=batch["hidden_mask"],
    )
    lm_ce = _lm_ce_loss(
        student_logits=outputs.logits,
        token_ids=batch["token_ids"],
        token_mask=batch["token_mask"],
    )
    total = coc_kl + config.alpha * hidden_align + config.beta * lm_ce

    for name, value in (
        ("coc_kl", coc_kl),
        ("hidden_align", hidden_align),
        ("lm_ce", lm_ce),
        ("total", total),
    ):
        _check_finite(name, value)

    return Stage2LossOutput(total=total, coc_kl=coc_kl, hidden_align=hidden_align, lm_ce=lm_ce)
