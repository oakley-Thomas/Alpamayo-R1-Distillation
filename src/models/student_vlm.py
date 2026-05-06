"""Student Qwen VLM wrapper and hidden-state adapter for Stage 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from src.losses.stage2 import Stage2ModelOutput

QLORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class StudentVLMConfig:
    """Configuration for loading the real Qwen2.5-VL Stage 2 student."""

    backbone_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05


class HiddenStateAdapter(nn.Module):
    """Project student hidden states into the teacher Action-Expert conditioning space."""

    def __init__(self, student_hidden_dim: int, teacher_hidden_dim: int) -> None:
        """Initialize the adapter.

        Args:
            student_hidden_dim: Last-layer student VLM hidden dimension.
            teacher_hidden_dim: Teacher hidden dimension consumed by the Action Expert.
        """
        super().__init__()
        self.proj = nn.Linear(student_hidden_dim, teacher_hidden_dim)
        self.norm = nn.LayerNorm(teacher_hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states.

        Args:
            hidden_states: Student hidden states, shape (B, T, D_student).

        Returns:
            Teacher-space hidden states, shape (B, T, D_teacher).
        """
        return self.norm(self.proj(hidden_states.float()))


class StudentVLM(nn.Module):
    """Qwen2.5-VL backbone plus Stage 2 hidden-state adapter."""

    def __init__(
        self,
        backbone: nn.Module,
        student_hidden_dim: int,
        teacher_hidden_dim: int,
    ) -> None:
        """Initialize from a backbone module.

        Args:
            backbone: Causal VLM module returning logits and hidden states.
            student_hidden_dim: Last-layer student hidden dimension.
            teacher_hidden_dim: Teacher hidden dimension read from the dump.
        """
        super().__init__()
        self.backbone = backbone
        self.hidden_adapter = HiddenStateAdapter(student_hidden_dim, teacher_hidden_dim)

    @classmethod
    def from_pretrained(
        cls,
        config: StudentVLMConfig,
        teacher_hidden_dim: int,
    ) -> StudentVLM:
        """Load Qwen2.5-VL-3B in 4-bit NF4 with QLoRA adapters.

        Args:
            config: QLoRA and model-loading options.
            teacher_hidden_dim: Adapter output dimension from the teacher dump.

        Returns:
            Stage 2 student model with a frozen vision tower and trainable QLoRA
            plus hidden adapter.
        """
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from transformers import AutoModelForVision2Seq, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "Loading the real Stage 2 model requires transformers, peft, and bitsandbytes"
            ) from exc

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        backbone = AutoModelForVision2Seq.from_pretrained(
            config.backbone_name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        )
        backbone = prepare_model_for_kbit_training(backbone, use_gradient_checkpointing=True)
        _freeze_vision_modules(backbone)
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=list(QLORA_TARGET_MODULES),
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        backbone = get_peft_model(backbone, lora_config)
        student_hidden_dim = _infer_hidden_dim(backbone)
        return cls(
            backbone=backbone,
            student_hidden_dim=student_hidden_dim,
            teacher_hidden_dim=teacher_hidden_dim,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        hidden_position_mask: torch.Tensor | None = None,
        **model_kwargs: Any,
    ) -> Stage2ModelOutput:
        """Run the student VLM and adapt selected hidden states.

        Args:
            input_ids: Token IDs, shape (B, L).
            attention_mask: Optional attention mask, shape (B, L).
            hidden_position_mask: Optional boolean mask selecting Action-Expert
                conditioning positions from the final decoder layer, shape (B, L).
            **model_kwargs: Extra model-specific tensors such as image inputs.

        Returns:
            Student logits, shape (B, L, V), and adapted hidden states, shape
            (B, T_cond, D_teacher).
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **model_kwargs,
        )
        hidden_states = _last_hidden_state(outputs)
        selected_hidden = _select_hidden_positions(hidden_states, hidden_position_mask)
        return Stage2ModelOutput(
            logits=outputs.logits,
            adapted_hidden_states=self.hidden_adapter(selected_hidden),
        )


def _last_hidden_state(outputs: Any) -> torch.Tensor:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None or len(hidden_states) == 0:
        raise RuntimeError("Student backbone must return hidden_states for Stage 2 alignment")
    last_hidden = hidden_states[-1]
    if not isinstance(last_hidden, torch.Tensor):
        raise TypeError("Last hidden state must be a torch.Tensor")
    return last_hidden


def _select_hidden_positions(
    hidden_states: torch.Tensor,
    hidden_position_mask: torch.Tensor | None,
) -> torch.Tensor:
    if hidden_position_mask is None:
        return hidden_states
    if hidden_position_mask.shape != hidden_states.shape[:2]:
        raise ValueError("hidden_position_mask must have shape (B, L)")

    selected: list[torch.Tensor] = []
    lengths: list[int] = []
    for batch_idx in range(hidden_states.shape[0]):
        values = hidden_states[batch_idx][hidden_position_mask[batch_idx]]
        if values.shape[0] == 0:
            raise ValueError("Each sample must select at least one hidden position")
        selected.append(values)
        lengths.append(int(values.shape[0]))

    max_len = max(lengths)
    padded = hidden_states.new_zeros((hidden_states.shape[0], max_len, hidden_states.shape[-1]))
    for batch_idx, values in enumerate(selected):
        padded[batch_idx, : values.shape[0]] = values
    return padded


def _freeze_vision_modules(model: nn.Module) -> None:
    for attr_name in ("visual", "vision_model", "vision_tower"):
        module = getattr(model, attr_name, None)
        if isinstance(module, nn.Module):
            for parameter in module.parameters():
                parameter.requires_grad = False


def _infer_hidden_dim(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    for source in (text_config, config):
        hidden_size = getattr(source, "hidden_size", None)
        if isinstance(hidden_size, int):
            return hidden_size
    raise RuntimeError("Could not infer student hidden size from backbone config")
