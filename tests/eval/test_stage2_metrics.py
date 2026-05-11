"""Tests for Stage 2 evaluation metrics."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.data.teacher_dump import TeacherDumpDataset
from src.eval.stage2 import (
    evaluate_stage2_model,
    hidden_cosine_similarity,
    qwen_coc_token_accuracy,
    trace_is_parseable,
)
from src.losses.stage2 import Stage2ModelOutput
from src.train.config import load_stage2_config


def test_qwen_coc_token_accuracy_uses_valid_tokens() -> None:
    logits = torch.zeros((1, 3, 8))
    logits[0, 0, 2] = 1.0
    logits[0, 1, 4] = 1.0
    logits[0, 2, 7] = 1.0
    lm_token_ids = torch.tensor([[2, 5, 7]])
    token_mask = torch.tensor([[True, True, False]])
    assert qwen_coc_token_accuracy(logits, lm_token_ids, token_mask).item() == 0.5


def test_hidden_cosine_similarity_is_one_for_equal_tensors() -> None:
    hidden = torch.randn(1, 2, 4)
    mask = torch.ones((1, 2), dtype=torch.bool)
    assert torch.allclose(hidden_cosine_similarity(hidden, hidden.clone(), mask), torch.tensor(1.0))


def test_trace_parseability_accepts_text_and_rejects_bad_json() -> None:
    assert trace_is_parseable("yield to pedestrian")
    assert trace_is_parseable('{"action": "yield"}')
    assert not trace_is_parseable('{"action": "yield"')
    assert not trace_is_parseable("   ")


class FakeStage2EvalModel(nn.Module):
    """Deterministic model for Stage 2 eval runner tests."""

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_position_mask: torch.Tensor,
        logit_position_mask: torch.Tensor,
        **_kwargs: object,
    ) -> Stage2ModelOutput:
        """Return logits matching the fixture and hidden states matching the dump."""
        token_count = int(logit_position_mask.sum().item())
        hidden_count = int(hidden_position_mask.sum().item())
        logits = torch.zeros((1, token_count, 32))
        label_ids = input_ids[0][logit_position_mask[0]]
        for index, token_id in enumerate(label_ids):
            logits[0, index, int(token_id.item())] = 1.0
        hidden = torch.arange(hidden_count * 4, dtype=torch.float32).reshape(1, hidden_count, 4)
        return Stage2ModelOutput(logits=logits, adapted_hidden_states=hidden)


def test_evaluate_stage2_model_reports_acceptance_metrics(mini_dump: tuple[Path, Path]) -> None:
    dump_root, split_file = mini_dump
    dataset = TeacherDumpDataset(dump_root, split_file)
    config = load_stage2_config("configs/stage2.yaml")
    report = evaluate_stage2_model(
        model=FakeStage2EvalModel(),
        dataset=dataset,
        config=config,
        device=torch.device("cpu"),
        processor=None,
    )
    assert report.num_clips == 2
    assert report.coc_token_accuracy == 1.0
    assert report.trace_parseability_rate == 1.0
    assert report.passes_coc_token_accuracy
