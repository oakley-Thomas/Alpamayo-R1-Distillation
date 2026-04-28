from __future__ import annotations

from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader

from alpamayo_distill.config import load_yaml, merge_dicts
from alpamayo_distill.data.dataset import TeacherRolloutDataset
from alpamayo_distill.data.tokenizer_utils import make_attention_mask, pad_token_sequences
from alpamayo_distill.losses.combined import combined_loss
from alpamayo_distill.models.student_vla import StudentVLA


def _pad_topk_tensors(samples: list[dict], key: str, pad_value: int) -> torch.Tensor:
    max_len = max(sample[key].size(0) for sample in samples)
    topk = samples[0][key].size(1)
    dtype = samples[0][key].dtype
    output = torch.full((len(samples), max_len, topk), pad_value, dtype=dtype)
    for row, sample in enumerate(samples):
        seq = sample[key]
        output[row, : seq.size(0)] = seq
    return output


def collate_batch(samples: list[dict]) -> dict[str, torch.Tensor]:
    tokens = pad_token_sequences([sample["teacher_tokens"] for sample in samples], pad_value=0)
    topk_idx = _pad_topk_tensors(samples, "teacher_topk_indices", pad_value=0)
    topk_logits = _pad_topk_tensors(samples, "teacher_topk_logits", pad_value=0)
    return {
        "cameras": torch.stack([sample["cameras"] for sample in samples], dim=0),
        "egomotion": torch.stack([sample["egomotion"] for sample in samples], dim=0),
        "text": tokens,
        "attention_mask": make_attention_mask(tokens),
        "teacher_topk_indices": topk_idx,
        "teacher_topk_logits": topk_logits,
        "teacher_trajectory": torch.stack([sample["teacher_trajectory"] for sample in samples], dim=0),
        "groundtruth_trajectory": torch.stack([sample["groundtruth_trajectory"] for sample in samples], dim=0),
    }


def load_training_bundle(
    student_cfg_path: str = "configs/student.yaml",
    distill_cfg_path: str = "configs/distill.yaml",
    data_cfg_path: str = "configs/data.yaml",
) -> tuple[StudentVLA, dict, DataLoader]:
    student_cfg = load_yaml(student_cfg_path)
    distill_cfg = load_yaml(distill_cfg_path)
    data_cfg = load_yaml(data_cfg_path)
    model = StudentVLA(student_cfg)
    dataset = TeacherRolloutDataset(Path(data_cfg["rollout"]["cache_dir"]))
    dataloader = DataLoader(
        dataset,
        batch_size=int(data_cfg["dataloader"]["batch_size_per_gpu"]),
        shuffle=True,
        collate_fn=collate_batch,
    )
    cfg = merge_dicts(student_cfg, distill_cfg)
    cfg = merge_dicts(cfg, data_cfg)
    return model, cfg, dataloader


def run_single_epoch(model: StudentVLA, cfg: dict, dataloader: DataLoader, lr: float) -> dict[str, float]:
    if len(dataloader.dataset) == 0:
        raise RuntimeError("Teacher rollout cache is empty; run rollout first")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    running = {"total": 0.0, "kd": 0.0, "trajectory": 0.0, "groundtruth": 0.0}
    for batch in dataloader:
        outputs = model(batch)
        losses = combined_loss(outputs, batch, cfg)
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()
        for key in running:
            running[key] += float(losses[key].detach().item())
    steps = len(dataloader)
    return {key: value / steps for key, value in running.items()}


def save_stage_artifacts(model: StudentVLA, stage_name: str, metrics: dict[str, float], root: str = "ckpts") -> None:
    out_dir = Path(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / f"{stage_name}.pt")
    (out_dir / f"{stage_name}.metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
