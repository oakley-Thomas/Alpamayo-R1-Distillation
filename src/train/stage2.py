"""Stage 2 training skeleton for VLM backbone distillation."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from src.data.teacher_dump import TeacherDumpDataset, collate_teacher_examples
from src.losses.stage2 import Stage2LossConfig, compute_stage2_loss
from src.models.student_vlm import StudentVLM, StudentVLMConfig
from src.train.config import Stage2Config, load_stage2_config
from src.utils.seed import set_seed


class Stage2Processor(Protocol):
    """Runtime protocol for Hugging Face processors used by Stage 2."""

    tokenizer: Any

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str: ...

    def __call__(self, **kwargs: Any) -> Mapping[str, Any]: ...


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
            compute_dtype=resolve_stage2_compute_dtype(config),
            gradient_checkpointing=config.training.gradient_checkpointing,
        ),
        teacher_hidden_dim=teacher_hidden_dim,
    )


def resolve_stage2_compute_dtype(config: Stage2Config) -> torch.dtype:
    """Resolve the Stage 2 model compute dtype from config and hardware support."""
    if not config.training.bf16:
        return torch.float16
    if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "Stage 2 config requests bf16, but this CUDA device does not support bf16. "
            "Set training.bf16 to false to load Qwen with fp16 on this GPU."
        )
    return torch.bfloat16


def build_stage2_processor(config: Stage2Config) -> Stage2Processor:
    """Load the Hugging Face processor that turns dump frames into VLM tensors."""
    try:
        transformers = importlib.import_module("transformers")
        auto_processor = transformers.AutoProcessor
    except ImportError as exc:
        raise RuntimeError("Stage 2 image preparation requires transformers.AutoProcessor") from exc
    processor = auto_processor.from_pretrained(config.model.processor_name)
    return cast(Stage2Processor, processor)


def _load_rgb_frame(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def select_stage2_frame_paths(frame_paths: list[Path], max_frames: int) -> list[Path]:
    """Select a bounded, evenly spaced set of frame paths for Stage 2 VLM input."""
    if max_frames <= 0:
        raise ValueError("Stage 2 max_frames must be positive")
    if len(frame_paths) <= max_frames:
        return frame_paths
    if max_frames == 1:
        return [frame_paths[0]]

    last_index = len(frame_paths) - 1
    return [
        frame_paths[round(index * last_index / (max_frames - 1))] for index in range(max_frames)
    ]


def _processor_outputs_for_example(
    processor: Stage2Processor,
    frame_paths: list[Path],
    prompt: str,
    max_frames: int,
) -> dict[str, torch.Tensor]:
    selected_paths = select_stage2_frame_paths(frame_paths, max_frames)
    images = [_load_rgb_frame(path) for path in selected_paths]
    content: list[dict[str, Any]] = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    raw_outputs = processor(
        text=[text],
        images=images,
        padding=True,
        return_tensors="pt",
    )
    outputs: dict[str, torch.Tensor] = {}
    for key, value in raw_outputs.items():
        if isinstance(value, torch.Tensor):
            outputs[key] = value
    if "input_ids" not in outputs:
        raise RuntimeError("Stage 2 processor did not return input_ids")
    if "attention_mask" not in outputs:
        outputs["attention_mask"] = torch.ones_like(outputs["input_ids"])
    return outputs


def _conditioning_meta_at(batch: Mapping[str, Any], batch_idx: int) -> Mapping[str, Any] | None:
    metas = batch.get("conditioning_meta")
    if not isinstance(metas, list):
        return None
    metas_list = cast(list[object], metas)
    meta = metas_list[batch_idx]
    if meta is None:
        return None
    if not isinstance(meta, dict):
        raise TypeError("conditioning_meta entries must be mappings or None")
    return cast(Mapping[str, Any], meta)


def _batch_frame_paths_at(batch: Mapping[str, Any], batch_idx: int) -> list[Path]:
    frame_path_batches = batch.get("frame_paths")
    if not isinstance(frame_path_batches, list):
        raise TypeError("batch frame_paths must be a list")
    frame_path_batches_list = cast(list[object], frame_path_batches)
    frame_paths = frame_path_batches_list[batch_idx]
    if not isinstance(frame_paths, list):
        raise TypeError("each batch frame_paths entry must be a list")
    paths = cast(list[object], frame_paths)
    if not all(isinstance(path, (str, Path)) for path in paths):
        raise TypeError("frame path entries must be strings or Paths")
    return [Path(path) for path in paths if isinstance(path, (str, Path))]


def _batch_coc_text_at(batch: Mapping[str, Any], batch_idx: int) -> str:
    coc_texts = batch.get("coc_text")
    if not isinstance(coc_texts, list):
        raise TypeError("batch coc_text must be a list")
    texts = cast(list[object], coc_texts)
    text = texts[batch_idx]
    if not isinstance(text, str):
        raise TypeError("coc_text entries must be strings")
    if not text.strip():
        raise ValueError("coc_text entries must be non-empty")
    return text


def _tokenize_coc_text(tokenizer: Any, text: str) -> torch.Tensor:
    if tokenizer is None or not callable(tokenizer):
        raise RuntimeError("Stage 2 Qwen-token CE requires processor.tokenizer")
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    if not isinstance(encoded, Mapping):
        raise RuntimeError("Stage 2 tokenizer did not return a mapping")
    encoded_mapping = cast(Mapping[str, object], encoded)
    input_ids_obj = encoded_mapping.get("input_ids")
    if not isinstance(input_ids_obj, torch.Tensor):
        raise RuntimeError("Stage 2 tokenizer did not return input_ids")
    if input_ids_obj.ndim == 2:
        if input_ids_obj.shape[0] != 1:
            raise ValueError("Stage 2 expects one CoC text per tokenizer call")
        input_ids = input_ids_obj.squeeze(0)
    elif input_ids_obj.ndim == 1:
        input_ids = input_ids_obj
    else:
        raise ValueError("Stage 2 tokenizer input_ids must have shape (L,) or (1, L)")
    if input_ids.numel() == 0:
        raise ValueError("Stage 2 tokenizer produced no CoC tokens")
    return input_ids.to(dtype=torch.long)


def _hidden_position_mask_for_retokenized_replay(
    *,
    sequence_length: int,
    teacher_hidden_length: int,
    conditioning_meta: Mapping[str, Any] | None,
) -> torch.Tensor:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if teacher_hidden_length <= 0:
        raise ValueError("teacher_hidden_length must be positive")

    if conditioning_meta is not None:
        offset = conditioning_meta.get("traj_future_start_offset")
        if not isinstance(offset, int) or offset < 0:
            raise ValueError(
                "conditioning metadata traj_future_start_offset must be a non-negative int"
            )
        if offset != teacher_hidden_length:
            raise ValueError(
                "teacher hidden length must match traj_future_start_offset; "
                f"got {teacher_hidden_length} and {offset}"
            )

    if teacher_hidden_length > sequence_length:
        raise ValueError(
            "retokenized replay sequence is shorter than teacher hidden states; "
            f"got sequence length {sequence_length} and hidden length {teacher_hidden_length}"
        )

    mask = torch.zeros(sequence_length, dtype=torch.bool)
    mask[:teacher_hidden_length] = True
    return mask


def _causal_lm_position_mask(
    *,
    sequence_length: int,
    prompt_length: int,
    target_token_count: int,
) -> torch.Tensor:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive for causal LM replay")
    if target_token_count <= 0:
        raise ValueError("target_token_count must be positive")

    logit_start = prompt_length - 1
    logit_stop = logit_start + target_token_count
    if logit_stop > sequence_length:
        raise ValueError(
            "retokenized replay is too short for causal LM labels; "
            f"need position {logit_stop - 1}, length is {sequence_length}"
        )

    mask = torch.zeros(sequence_length, dtype=torch.bool)
    mask[logit_start:logit_stop] = True
    return mask


def prepare_stage2_model_inputs(
    batch: Mapping[str, Any],
    config: Stage2Config,
    device: torch.device,
    processor: Stage2Processor | None,
) -> dict[str, Any]:
    """Build student VLM inputs and alignment masks for one Stage 2 batch.

    The trainer uses batch size 1 because each clip carries a variable number
    of video frames. This function still returns batched tensors so the model
    path remains the same as the test path.
    """
    token_ids = batch["token_ids"]
    token_mask = batch["token_mask"]
    hidden_mask = batch["hidden_mask"]
    if not isinstance(token_ids, torch.Tensor):
        raise TypeError("batch token_ids must be a tensor")
    if not isinstance(token_mask, torch.Tensor):
        raise TypeError("batch token_mask must be a tensor")
    if not isinstance(hidden_mask, torch.Tensor):
        raise TypeError("batch hidden_mask must be a tensor")
    if token_ids.shape[0] != 1:
        raise ValueError("Stage 2 VLM input preparation expects one clip per batch")

    raw_valid_tokens = token_ids[0][token_mask[0]]
    conditioning_meta = _conditioning_meta_at(batch, 0)
    if processor is None:
        lm_token_ids = raw_valid_tokens
        input_ids = lm_token_ids.unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        model_kwargs: dict[str, torch.Tensor] = {}
        hidden_position_mask = torch.ones(input_ids.shape[1], dtype=torch.bool)
        logit_position_mask = torch.ones(input_ids.shape[1], dtype=torch.bool)
    else:
        processor_outputs = _processor_outputs_for_example(
            processor=processor,
            frame_paths=_batch_frame_paths_at(batch, 0),
            prompt=config.data.image_prompt,
            max_frames=config.data.max_frames,
        )
        prompt_input_ids = processor_outputs.pop("input_ids")
        prompt_attention_mask = processor_outputs.pop("attention_mask")
        lm_token_ids = _tokenize_coc_text(processor.tokenizer, _batch_coc_text_at(batch, 0))
        generated_attention_mask = torch.ones((1, lm_token_ids.shape[0]), dtype=torch.long)
        input_ids = torch.cat([prompt_input_ids, lm_token_ids.unsqueeze(0)], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, generated_attention_mask], dim=1)
        model_kwargs = processor_outputs
        hidden_position_mask = _hidden_position_mask_for_retokenized_replay(
            sequence_length=int(input_ids.shape[1]),
            teacher_hidden_length=int(hidden_mask[0].sum().item()),
            conditioning_meta=conditioning_meta,
        )
        logit_position_mask = _causal_lm_position_mask(
            sequence_length=int(input_ids.shape[1]),
            prompt_length=int(prompt_input_ids.shape[1]),
            target_token_count=int(lm_token_ids.shape[0]),
        )

    lm_token_mask = torch.ones_like(lm_token_ids, dtype=torch.bool)
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "hidden_position_mask": hidden_position_mask.unsqueeze(0).to(device),
        "logit_position_mask": logit_position_mask.unsqueeze(0).to(device),
        "lm_token_ids": lm_token_ids.unsqueeze(0).to(device),
        "lm_token_mask": lm_token_mask.unsqueeze(0).to(device),
        "model_kwargs": {key: value.to(device) for key, value in model_kwargs.items()},
    }


def save_stage2_artifacts(model: StudentVLM, output_dir: str | Path) -> None:
    """Save the LoRA-backed VLM adapter and hidden-state adapter bundle."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    save_pretrained = getattr(model.backbone, "save_pretrained", None)
    if callable(save_pretrained):
        save_pretrained(destination / "lora_adapter")
    else:
        torch.save(model.backbone.state_dict(), destination / "backbone_state.pt")
    torch.save(model.hidden_adapter.state_dict(), destination / "hidden_adapter.pt")
    metadata = {
        "format": "stage2_student_vlm",
        "contains": ["lora_adapter", "hidden_adapter"],
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def load_stage2_artifacts(model: StudentVLM, output_dir: str | Path) -> None:
    """Load a saved Stage 2 LoRA adapter and hidden-state adapter bundle.

    Args:
        model: Fresh Stage 2 student with the same backbone and adapter dims.
        output_dir: Directory produced by :func:`save_stage2_artifacts`.

    Raises:
        FileNotFoundError: If required adapter files are missing.
        RuntimeError: If a LoRA adapter exists but the backbone cannot load it.
    """
    source = Path(output_dir)
    hidden_adapter_path = source / "hidden_adapter.pt"
    if not hidden_adapter_path.is_file():
        raise FileNotFoundError(f"missing Stage 2 hidden adapter: {hidden_adapter_path}")

    state_obj = torch.load(hidden_adapter_path, map_location="cpu")
    if not isinstance(state_obj, dict):
        raise RuntimeError(f"{hidden_adapter_path} did not contain a state dict")
    model.hidden_adapter.load_state_dict(cast(dict[str, torch.Tensor], state_obj))

    lora_dir = source / "lora_adapter"
    if lora_dir.is_dir():
        load_adapter = getattr(model.backbone, "load_adapter", None)
        if not callable(load_adapter):
            raise RuntimeError("saved LoRA adapter exists but backbone cannot load adapters")
        load_adapter(str(lora_dir), adapter_name="stage2", is_trainable=False)
        set_adapter = getattr(model.backbone, "set_adapter", None)
        if callable(set_adapter):
            set_adapter("stage2")
        return

    backbone_state_path = source / "backbone_state.pt"
    if backbone_state_path.is_file():
        backbone_state_obj = torch.load(backbone_state_path, map_location="cpu")
        if not isinstance(backbone_state_obj, dict):
            raise RuntimeError(f"{backbone_state_path} did not contain a state dict")
        model.backbone.load_state_dict(cast(dict[str, torch.Tensor], backbone_state_obj))
        return

    raise FileNotFoundError(f"missing Stage 2 LoRA adapter or backbone state under {source}")


def cache_student_hidden_states(
    model: StudentVLM,
    dataset: TeacherDumpDataset,
    output_dir: str | Path,
    device: torch.device,
    config: Stage2Config,
    processor: Stage2Processor | None,
) -> None:
    """Write adapted student hidden states for every clip in a dataset.

    Args:
        model: Stage 2 student in eval mode.
        dataset: Teacher dump dataset whose CoC text drives Qwen-token replay.
        output_dir: Destination for ``<clip_id>.npy`` cache files.
        device: Device used for model inference.
        config: Stage 2 config controlling image preparation.
        processor: Optional processor for frame-backed VLM inputs.
    """
    cache_dir = Path(output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for example in dataset:
            batch = collate_teacher_examples([example])
            prepared = prepare_stage2_model_inputs(batch, config, device, processor)
            outputs = model(
                input_ids=prepared["input_ids"],
                attention_mask=prepared["attention_mask"],
                hidden_position_mask=prepared["hidden_position_mask"],
                logit_position_mask=prepared["logit_position_mask"],
                **prepared["model_kwargs"],
            )
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
    processor = build_stage2_processor(config)
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
            prepared = prepare_stage2_model_inputs(batch, config, device, processor)
            tensor_batch["lm_token_ids"] = prepared["lm_token_ids"]
            tensor_batch["lm_token_mask"] = prepared["lm_token_mask"]
            outputs = model(
                input_ids=prepared["input_ids"],
                attention_mask=prepared["attention_mask"],
                hidden_position_mask=prepared["hidden_position_mask"],
                logit_position_mask=prepared["logit_position_mask"],
                **prepared["model_kwargs"],
            )
            loss: torch.Tensor = compute_stage2_loss(tensor_batch, outputs, loss_config).total
            scaled_loss: torch.Tensor = loss / config.training.gradient_accumulation_steps
            torch.autograd.backward([scaled_loss])
            if (step_idx + 1) % config.training.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    save_stage2_artifacts(model, config.outputs.student_vlm_dir)
    for split_file in (config.data.train_split, config.data.val_split, config.data.test_split):
        cache_dataset = TeacherDumpDataset(
            root=config.data.teacher_dump_root,
            split_file=split_file,
            include_kv_cache=config.data.include_kv_cache,
        )
        cache_student_hidden_states(
            model,
            cache_dataset,
            config.outputs.hidden_cache_dir,
            device,
            config,
            processor,
        )
