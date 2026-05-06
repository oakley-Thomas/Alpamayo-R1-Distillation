"""Evaluation helpers for distillation stages."""

from __future__ import annotations

from src.eval.stage2 import (
    Stage2EvalReport,
    coc_top1_agreement,
    evaluate_stage2_model,
    hidden_cosine_similarity,
    run_stage2_evaluation,
    trace_is_parseable,
    write_stage2_eval_report,
)

__all__ = [
    "Stage2EvalReport",
    "coc_top1_agreement",
    "evaluate_stage2_model",
    "hidden_cosine_similarity",
    "run_stage2_evaluation",
    "trace_is_parseable",
    "write_stage2_eval_report",
]
