from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

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


class StudentVLA(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
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
        self.backbone = StubBackbone(
            vocab_size=int(config["backbone"]["vocab_size"]),
            hidden_dim=hidden_dim,
            num_layers=int(config["backbone"]["num_layers"]),
            num_heads=int(config["backbone"]["num_heads"]),
        )
        output_cfg = config["action_decoder"]["output"]
        output_steps = int(output_cfg["horizon_seconds"] * output_cfg["waypoints_per_second"])
        self.action_decoder = build_action_decoder(config["action_decoder"], hidden_dim=hidden_dim, output_steps=output_steps)

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
