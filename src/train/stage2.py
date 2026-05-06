"""Stage 2 training skeleton for VLM backbone distillation."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.data.teacher_dump import TeacherDumpDataset, collate_teacher_examples
from src.losses.stage2 import Stage2LossConfig, compute_stage2_loss
from src.models.student_vlm import StudentVLM, StudentVLMConfig
from src.train.config import Stage2Config, load_stage2_config
from src.utils.seed import set_seed


def _run_text_command(args: list[str]) -> str:
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def write_reproducibility_files(config: Stage2Config, config_path: Path, output_dir: Path) -> None:
    """Write config, git SHA, and environment metadata for a Stage 2 run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False)
    git_sha = _run_text_command(["git", "rev-parse", "HEAD"])
    (output_dir / "git_sha.txt").write_text(git_sha + "\n", encoding="utf-8")
    env_text = _run_text_command([sys.executable, "-m", "pip", "freeze"])
    (output_dir / "env.txt").write_text(env_text + "\n", encoding="utf-8")
    (output_dir / "source_config_path.txt").write_text(str(config_path) + "\n", encoding="utf-8")


def trainable_lora_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    """Yield trainable non-adapter parameters, normally QLoRA weights."""
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and not name.startswith("hidden_adapter."):
            yield parameter


def adapter_parameters(model: StudentVLM) -> Iterable[nn.Parameter]:
    """Yield trainable hidden adapter parameters."""
    yield from model.hidden_adapter.parameters()


def build_stage2_optimizer(model: StudentVLM, config: Stage2Config) -> torch.optim.Optimizer:
    """Build the required paged AdamW optimizer with separate parameter groups."""
    try:
        from bitsandbytes.optim.adamw import PagedAdamW8bit
    except ImportError as exc:
        raise RuntimeError("Stage 2 requires bitsandbytes.optim.PagedAdamW8bit") from exc

    return PagedAdamW8bit(
        [
            {
                "params": list(trainable_lora_parameters(model)),
                "lr": config.optimizer.lora_lr,
                "weight_decay": config.optimizer.weight_decay,
            },
            {
                "params": list(adapter_parameters(model)),
                "lr": config.optimizer.adapter_lr,
                "weight_decay": config.optimizer.weight_decay,
            },
        ]
    )


def _infer_teacher_hidden_dim(dataset: TeacherDumpDataset) -> int:
    if len(dataset.manifests) == 0:
        raise ValueError("Stage 2 training split is empty")
    return dataset.manifests[0].hidden_shape[1]


def build_stage2_model(config: Stage2Config, teacher_hidden_dim: int) -> StudentVLM:
    """Build the real Stage 2 student model from config."""
    return StudentVLM.from_pretrained(
        StudentVLMConfig(
            backbone_name=config.model.backbone_name,
            lora_rank=config.model.lora_rank,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,
        ),
        teacher_hidden_dim=teacher_hidden_dim,
    )


def cache_student_hidden_states(
    model: StudentVLM,
    dataset: TeacherDumpDataset,
    output_dir: str | Path,
    device: torch.device,
) -> None:
    """Write adapted student hidden states for every clip in a dataset.

    Args:
        model: Stage 2 student in eval mode.
        dataset: Teacher dump dataset whose token IDs drive the replay.
        output_dir: Destination for ``<clip_id>.npy`` cache files.
        device: Device used for model inference.
    """
    cache_dir = Path(output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for example in dataset:
            token_ids = example["token_ids"].unsqueeze(0).to(device)
            outputs = model(input_ids=token_ids)
            hidden = outputs.adapted_hidden_states.squeeze(0).detach().cpu().to(torch.float16)
            np.save(cache_dir / f"{example['clip_id']}.npy", hidden.numpy())


def run_stage2_training(config_path: str | Path) -> None:
    """Run the Stage 2 training skeleton.

    This wires the real data/model/loss path and intentionally requires CUDA
    when configured, because silently training the 4-bit VLM on CPU would hide a
    setup error rather than produce a useful run.
    """
    config_file = Path(config_path)
    config = load_stage2_config(config_file)
    set_seed(config.training.seed)
    if config.training.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("Stage 2 config requires CUDA, but torch.cuda.is_available() is false")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = TeacherDumpDataset(
        root=config.data.teacher_dump_root,
        split_file=config.data.train_split,
        include_kv_cache=config.data.include_kv_cache,
    )
    teacher_hidden_dim = config.model.teacher_hidden_dim or _infer_teacher_hidden_dim(train_dataset)
    model = build_stage2_model(config, teacher_hidden_dim=teacher_hidden_dim).to(device)
    optimizer = build_stage2_optimizer(model, config)
    loss_config = Stage2LossConfig(
        alpha=config.loss.alpha,
        beta=config.loss.beta,
        temperature=config.loss.temperature,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_teacher_examples,
    )
    write_reproducibility_files(config, config_file, Path(config.outputs.root))

    model.train()
    optimizer.zero_grad(set_to_none=True)
    for _epoch in range(config.training.epochs):
        for step_idx, batch in enumerate(loader):
            tensor_batch = {
                key: value.to(device)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }
            outputs = model(input_ids=tensor_batch["token_ids"])
            loss: torch.Tensor = compute_stage2_loss(tensor_batch, outputs, loss_config).total
            scaled_loss: torch.Tensor = loss / config.training.gradient_accumulation_steps
            torch.autograd.backward([scaled_loss])
            if (step_idx + 1) % config.training.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    cache_dataset = TeacherDumpDataset(
        root=config.data.teacher_dump_root,
        split_file=config.data.val_split,
        include_kv_cache=config.data.include_kv_cache,
    )
    cache_student_hidden_states(model, cache_dataset, config.outputs.hidden_cache_dir, device)
