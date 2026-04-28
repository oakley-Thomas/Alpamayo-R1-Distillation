from __future__ import annotations

import torch
import torch.nn.functional as F


def kd_loss(
    student_logits: torch.Tensor,
    teacher_topk_logits: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    temperature: float = 2.0,
    mask: torch.Tensor | None = None,
    logit_scale: float = 1000.0,
) -> torch.Tensor:
    teacher_topk_logits = teacher_topk_logits.float() / logit_scale
    gathered = torch.gather(student_logits, dim=-1, index=teacher_topk_indices.long())
    teacher_probs = F.softmax(teacher_topk_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(gathered / temperature, dim=-1)
    token_loss = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1) * (temperature ** 2)
    if mask is not None:
        token_loss = token_loss * mask
        denom = mask.sum().clamp_min(1.0)
    else:
        denom = token_loss.numel()
    return token_loss.sum() / denom
