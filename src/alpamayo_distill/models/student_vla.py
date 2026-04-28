from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from alpamayo_distill.models.action_decoder import build_action_decoder
from alpamayo_distill.models.resampler import IdentityResampler
from alpamayo_distill.models.vision_encoder import SimpleVisionEncoder


class StubBackbone(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int, num_heads: int) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, prefix_tokens: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        token_hidden = self.token_embed(input_ids)
        hidden = torch.cat([prefix_tokens, token_hidden], dim=1) if prefix_tokens is not None else token_hidden
        encoded = self.encoder(hidden)
        text_hidden = encoded[:, -input_ids.size(1) :, :]
        logits = self.output_head(text_hidden)
        return logits, encoded


class EgomotionEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_tokens: int, output_dim: int) -> None:
        super().__init__()
        self.output_tokens = output_tokens
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_tokens * output_dim),
        )

    def forward(self, egomotion: torch.Tensor) -> torch.Tensor:
        pooled = egomotion.mean(dim=1)
        tokens = self.net(pooled)
        return tokens.view(egomotion.size(0), self.output_tokens, self.output_dim)


@dataclass
class StudentOutput:
    logits: torch.Tensor
    trajectory: torch.Tensor
    cot_text: list[str]
    hidden_states: torch.Tensor


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return mapping[dtype_name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype setting: {dtype_name}") from exc


class TransformersBackbone(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        backbone_cfg = config["backbone"]
        model_name = backbone_cfg["name"]
        dtype = _resolve_dtype(str(backbone_cfg["dtype"]))
        from_pretrained_kwargs = {
            "torch_dtype": dtype,
            "attn_implementation": backbone_cfg.get("attn_implementation", "sdpa"),
            "local_files_only": bool(backbone_cfg.get("local_files_only", False)),
            "trust_remote_code": bool(backbone_cfg.get("trust_remote_code", False)),
        }
        if bool(backbone_cfg.get("pretrained", True)):
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **from_pretrained_kwargs)
        else:
            raise RuntimeError("TransformersBackbone requires pretrained weights in the current implementation")
        hidden_size = self.model.config.text_config.hidden_size
        vocab_size = self.model.config.text_config.vocab_size
        config["backbone"]["hidden_dim"] = hidden_size
        config["backbone"]["vocab_size"] = vocab_size

    def forward(self, input_ids: torch.Tensor, prefix_tokens: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        input_embeds = self.model.get_input_embeddings()(input_ids)
        if prefix_tokens is not None:
            prefix_tokens = prefix_tokens.to(dtype=input_embeds.dtype, device=input_embeds.device)
            inputs_embeds = torch.cat([prefix_tokens, input_embeds], dim=1)
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=input_ids.device)
        else:
            inputs_embeds = input_embeds
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]
        logits = outputs.logits[:, -input_ids.size(1) :, :]
        return logits, hidden_states


class StudentVLA(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.processor = self._build_processor(config)
        hidden_dim = int(config["backbone"]["hidden_dim"])
        self.backbone_provider = str(config["backbone"].get("provider", "stub"))
        self.backbone = self._build_backbone(config)
        hidden_dim = int(config["backbone"]["hidden_dim"])
        self.vision_encoder = SimpleVisionEncoder(
            output_dim=hidden_dim,
            tokens_per_camera=int(config["vision_encoder"]["output_tokens_per_camera"]),
        )
        self.resampler = IdentityResampler()
        self.egomotion_encoder = EgomotionEncoder(
            input_dim=int(config["egomotion_encoder"]["input_dim"]),
            hidden_dim=int(config["egomotion_encoder"]["hidden_dim"]),
            output_tokens=int(config["egomotion_encoder"]["output_tokens"]),
            output_dim=hidden_dim,
        )
        output_cfg = config["action_decoder"]["output"]
        output_steps = int(output_cfg["horizon_seconds"] * output_cfg["waypoints_per_second"])
        self.action_decoder = build_action_decoder(config["action_decoder"], hidden_dim=hidden_dim, output_steps=output_steps)

    def _build_processor(self, config: dict[str, Any]):
        processor_cfg = config.get("processor", {})
        if processor_cfg.get("provider", "transformers") != "transformers":
            return None
        return AutoProcessor.from_pretrained(
            config["processor_id"],
            local_files_only=bool(processor_cfg.get("local_files_only", False)),
            trust_remote_code=bool(processor_cfg.get("trust_remote_code", False)),
        )

    def _build_backbone(self, config: dict[str, Any]) -> nn.Module:
        provider = str(config["backbone"].get("provider", "stub"))
        if provider == "transformers":
            return TransformersBackbone(config)
        if provider == "stub":
            return StubBackbone(
                vocab_size=int(config["backbone"]["vocab_size"]),
                hidden_dim=int(config["backbone"]["hidden_dim"]),
                num_layers=int(config["backbone"]["num_layers"]),
                num_heads=int(config["backbone"]["num_heads"]),
            )
        raise ValueError(f"Unsupported backbone provider: {provider}")

    def forward(self, batch: dict[str, torch.Tensor]) -> StudentOutput:
        cameras = batch["cameras"]
        if cameras.dim() == 6:
            cameras = cameras[:, :, -1, :, :, :]
        vision_tokens = self.resampler(self.vision_encoder(cameras))
        ego_tokens = self.egomotion_encoder(batch["egomotion"])
        prefix_tokens = torch.cat([ego_tokens, vision_tokens], dim=1)
        logits, hidden_states = self.backbone(batch["text"], prefix_tokens=prefix_tokens)
        trajectory = self.action_decoder(hidden_states)
        cot_text = [" ".join(map(str, row.tolist())) for row in batch["text"]]
        return StudentOutput(logits=logits, trajectory=trajectory, cot_text=cot_text, hidden_states=hidden_states)
