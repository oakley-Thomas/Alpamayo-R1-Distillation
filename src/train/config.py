"""Typed configuration loading for distillation training."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
import types
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml


T = TypeVar("T")


@dataclass(frozen=True)
class Stage2DataConfig:
    """Data locations for Stage 2."""

    teacher_dump_root: str
    train_split: str
    val_split: str
    include_kv_cache: bool = False


@dataclass(frozen=True)
class Stage2ModelConfig:
    """Student VLM loading options for Stage 2."""

    backbone_name: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    teacher_hidden_dim: int | None = None
    student_hidden_dim: int | None = None


@dataclass(frozen=True)
class Stage2OptimizerConfig:
    """Optimizer hyperparameters for Stage 2."""

    lora_lr: float
    adapter_lr: float
    weight_decay: float


@dataclass(frozen=True)
class Stage2TrainingConfig:
    """Training loop controls for Stage 2."""

    epochs: int
    gradient_accumulation_steps: int
    bf16: bool
    gradient_checkpointing: bool
    require_cuda: bool
    seed: int


@dataclass(frozen=True)
class Stage2OutputConfig:
    """Output locations for Stage 2 artifacts."""

    root: str
    student_vlm_dir: str
    hidden_cache_dir: str


@dataclass(frozen=True)
class Stage2LossYamlConfig:
    """Loss hyperparameters loaded from YAML."""

    alpha: float
    beta: float
    temperature: float


@dataclass(frozen=True)
class Stage2Config:
    """Complete Stage 2 training configuration."""

    data: Stage2DataConfig
    model: Stage2ModelConfig
    loss: Stage2LossYamlConfig
    optimizer: Stage2OptimizerConfig
    training: Stage2TrainingConfig
    outputs: Stage2OutputConfig


def _unwrap_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is types.UnionType:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _coerce_dataclass(cls: type[T], data: Any) -> T:
    if not isinstance(data, dict):
        raise ValueError(f"{cls.__name__} expects a mapping")
    kwargs: dict[str, Any] = {}
    type_hints = get_type_hints(cls)
    for field in fields(cls):
        if field.name not in data:
            raise ValueError(f"Missing config field {cls.__name__}.{field.name}")
        annotation = _unwrap_optional(type_hints[field.name])
        if hasattr(annotation, "__dataclass_fields__"):
            kwargs[field.name] = _coerce_dataclass(annotation, data[field.name])
        else:
            kwargs[field.name] = data[field.name]
    return cls(**kwargs)


def load_stage2_config(path: str | Path) -> Stage2Config:
    """Load a Stage 2 YAML config into typed dataclasses."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return _coerce_dataclass(Stage2Config, data)
