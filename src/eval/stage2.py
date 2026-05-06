"""Stage 2 VLM distillation metrics and acceptance runner."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from src.data.teacher_dump import TeacherDumpDataset, collate_teacher_examples
from src.losses.stage2 import Stage2ModelOutput
from src.train.config import Stage2Config, load_stage2_config
from src.train.stage2 import (
    Stage2Processor,
    build_stage2_model,
    build_stage2_processor,
    load_stage2_artifacts,
    prepare_stage2_model_inputs,
)

COC_TOP1_TARGET = 0.70
HIDDEN_COSINE_TARGET = 0.85
TRACE_PARSEABILITY_TARGET = 1.0


@dataclass(frozen=True)
class Stage2EvalReport:
    """Acceptance metrics for Stage 2 VLM distillation."""

    split_file: str
    num_clips: int
    coc_top1_agreement: float
    hidden_cosine_similarity: float
    trace_parseability_rate: float
    passes_coc_top1: bool
    passes_hidden_cosine: bool
    passes_trace_parseability: bool

    @property
    def passes_acceptance(self) -> bool:
        """Return whether all Stage 2 acceptance metrics pass."""
        return self.passes_coc_top1 and self.passes_hidden_cosine and self.passes_trace_parseability

    def to_json_dict(self) -> dict[str, bool | float | int | str]:
        """Return a JSON-serializable report dictionary."""
        data = asdict(self)
        data["passes_acceptance"] = self.passes_acceptance
        return cast(dict[str, bool | float | int | str], data)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=values.dtype)
    denominator = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denominator


def coc_top1_agreement(
    student_logits: torch.Tensor,
    teacher_top_k_token_ids: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute CoC token top-1 agreement against teacher top-k IDs.

    Args:
        student_logits: Student logits selected at CoC positions, shape (B, T, V).
        teacher_top_k_token_ids: Teacher top-k token IDs, shape (B, T, K).
        token_mask: Valid CoC token mask, shape (B, T).

    Returns:
        Mean top-1 agreement over valid CoC tokens, scalar tensor.
    """
    if student_logits.ndim != 3:
        raise ValueError("student_logits must have shape (B, T, V)")
    if teacher_top_k_token_ids.ndim != 3:
        raise ValueError("teacher_top_k_token_ids must have shape (B, T, K)")
    if student_logits.shape[:2] != teacher_top_k_token_ids.shape[:2]:
        raise ValueError("student and teacher token dimensions must match")
    if token_mask.shape != student_logits.shape[:2]:
        raise ValueError("token_mask must have shape (B, T)")

    student_top1 = student_logits.argmax(dim=-1)
    teacher_top1 = teacher_top_k_token_ids[..., 0]
    matches = (student_top1 == teacher_top1).to(dtype=student_logits.dtype)
    return _masked_mean(matches, token_mask)


def hidden_cosine_similarity(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    hidden_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute masked hidden-state cosine similarity.

    Args:
        student_hidden: Adapted student hidden states, shape (B, T, D_h).
        teacher_hidden: Teacher hidden states, shape (B, T, D_h).
        hidden_mask: Valid hidden-state mask, shape (B, T).

    Returns:
        Mean cosine similarity over valid hidden positions, scalar tensor.
    """
    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError("student_hidden and teacher_hidden must have matching shapes")
    if student_hidden.ndim != 3:
        raise ValueError("hidden tensors must have shape (B, T, D_h)")
    if hidden_mask.shape != student_hidden.shape[:2]:
        raise ValueError("hidden_mask must have shape (B, T)")

    similarities = F.cosine_similarity(student_hidden.float(), teacher_hidden.float(), dim=-1)
    return _masked_mean(similarities, hidden_mask)


def trace_is_parseable(text: str) -> bool:
    """Return whether a generated CoC trace is structurally parseable.

    Plain-text CoC traces are parseable when non-empty. Structured traces that
    begin with ``{`` or ``[`` must be valid JSON so malformed structured output
    is caught by the Stage 2 acceptance check.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if stripped[0] not in {"{", "["}:
        return True
    try:
        json.loads(stripped)
    except json.JSONDecodeError:
        return False
    return True


def decode_coc_tokens(processor: Stage2Processor | None, token_ids: torch.Tensor) -> str:
    """Decode predicted CoC token IDs for parseability checks.

    Args:
        processor: Optional Hugging Face processor with ``batch_decode`` or a
            tokenizer carrying ``decode``.
        token_ids: Predicted CoC token IDs, shape (T,).

    Returns:
        Decoded text, or a deterministic ID string when no decoder is available.
    """
    ids = token_ids.detach().cpu().to(dtype=torch.long)
    if processor is not None:
        batch_decode = getattr(processor, "batch_decode", None)
        if callable(batch_decode):
            decoded_obj = batch_decode(ids.unsqueeze(0), skip_special_tokens=True)
            if isinstance(decoded_obj, list) and decoded_obj and isinstance(decoded_obj[0], str):
                return decoded_obj[0]

        tokenizer = getattr(processor, "tokenizer", None)
        decode = getattr(tokenizer, "decode", None)
        if callable(decode):
            decoded = decode(ids, skip_special_tokens=True)
            if isinstance(decoded, str):
                return decoded

    return " ".join(str(int(token_id.item())) for token_id in ids.reshape(-1))


def _infer_teacher_hidden_dim(dataset: TeacherDumpDataset) -> int:
    if len(dataset.manifests) == 0:
        raise ValueError("Stage 2 eval split is empty")
    return dataset.manifests[0].hidden_shape[1]


def evaluate_stage2_model(
    *,
    model: nn.Module,
    dataset: TeacherDumpDataset,
    config: Stage2Config,
    device: torch.device,
    processor: Stage2Processor | None,
) -> Stage2EvalReport:
    """Evaluate a Stage 2 student on one split.

    Args:
        model: Stage 2 model returning :class:`Stage2ModelOutput`.
        dataset: Teacher dump split to evaluate.
        config: Stage 2 config controlling input preparation.
        device: Device used for model inference.
        processor: Optional processor for frame-backed VLM inputs and decoding.

    Returns:
        Stage 2 acceptance report.
    """
    model.eval()
    total_tokens = 0
    total_hidden_positions = 0
    weighted_coc_top1 = 0.0
    weighted_hidden_cosine = 0.0
    parseable_traces = 0

    with torch.no_grad():
        for example in dataset:
            batch = collate_teacher_examples([example])
            tensor_batch = {
                key: value.to(device)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }
            prepared = prepare_stage2_model_inputs(batch, config, device, processor)
            output_obj = model(
                input_ids=prepared["input_ids"],
                attention_mask=prepared["attention_mask"],
                hidden_position_mask=prepared["hidden_position_mask"],
                logit_position_mask=prepared["logit_position_mask"],
                **prepared["model_kwargs"],
            )
            outputs = cast(Stage2ModelOutput, output_obj)

            token_count = int(tensor_batch["token_mask"].sum().item())
            hidden_count = int(tensor_batch["hidden_mask"].sum().item())
            coc_top1 = coc_top1_agreement(
                outputs.logits,
                tensor_batch["top_k_token_ids"],
                tensor_batch["token_mask"],
            )
            hidden_cosine = hidden_cosine_similarity(
                outputs.adapted_hidden_states,
                tensor_batch["teacher_hidden_states"],
                tensor_batch["hidden_mask"],
            )
            predicted_ids = outputs.logits.argmax(dim=-1)[0, :token_count]
            decoded_trace = decode_coc_tokens(processor, predicted_ids)

            weighted_coc_top1 += float(coc_top1.item()) * token_count
            weighted_hidden_cosine += float(hidden_cosine.item()) * hidden_count
            total_tokens += token_count
            total_hidden_positions += hidden_count
            parseable_traces += int(trace_is_parseable(decoded_trace))

    if total_tokens == 0:
        raise ValueError("Stage 2 eval found no valid CoC tokens")
    if total_hidden_positions == 0:
        raise ValueError("Stage 2 eval found no valid hidden states")

    coc_value = weighted_coc_top1 / total_tokens
    hidden_value = weighted_hidden_cosine / total_hidden_positions
    parseability_value = parseable_traces / len(dataset)
    return Stage2EvalReport(
        split_file=str(dataset.split_file),
        num_clips=len(dataset),
        coc_top1_agreement=coc_value,
        hidden_cosine_similarity=hidden_value,
        trace_parseability_rate=parseability_value,
        passes_coc_top1=coc_value >= COC_TOP1_TARGET,
        passes_hidden_cosine=hidden_value >= HIDDEN_COSINE_TARGET,
        passes_trace_parseability=parseability_value >= TRACE_PARSEABILITY_TARGET,
    )


def run_stage2_evaluation(
    config_path: str | Path,
    split_file: str | Path | None = None,
    student_dir: str | Path | None = None,
) -> Stage2EvalReport:
    """Load Stage 2 artifacts and evaluate acceptance metrics.

    Args:
        config_path: Stage 2 YAML config path.
        split_file: Optional split override; defaults to the config val split.
        student_dir: Optional artifact override; defaults to the config output.

    Returns:
        Stage 2 acceptance report.
    """
    config = load_stage2_config(config_path)
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 2 eval requires CUDA for the real 4-bit VLM")
    device = torch.device("cuda")
    dataset = TeacherDumpDataset(
        root=config.data.teacher_dump_root,
        split_file=split_file or config.data.val_split,
        include_kv_cache=config.data.include_kv_cache,
    )
    model = build_stage2_model(config, teacher_hidden_dim=_infer_teacher_hidden_dim(dataset)).to(
        device
    )
    load_stage2_artifacts(model, student_dir or config.outputs.student_vlm_dir)
    processor = build_stage2_processor(config)
    return evaluate_stage2_model(
        model=model,
        dataset=dataset,
        config=config,
        device=device,
        processor=processor,
    )


def write_stage2_eval_report(report: Stage2EvalReport, output_path: str | Path) -> None:
    """Write a Stage 2 evaluation report as JSON."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report.to_json_dict(), indent=2) + "\n", encoding="utf-8")
