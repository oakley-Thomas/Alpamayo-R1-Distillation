from __future__ import annotations

from torch import nn
import torch


class FlowMatchingActionDecoder(nn.Module):
    def __init__(self, hidden_dim: int, output_steps: int, output_dim: int = 3) -> None:
        super().__init__()
        self.output_steps = output_steps
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_steps * output_dim),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        pooled = context.mean(dim=1)
        traj = self.net(pooled)
        return traj.view(context.size(0), self.output_steps, self.output_dim)


class MLPCodebookActionDecoder(nn.Module):
    def __init__(self, hidden_dim: int, output_steps: int, codebook_size: int, output_dim: int = 3) -> None:
        super().__init__()
        self.logits = nn.Linear(hidden_dim, codebook_size)
        self.codebook = nn.Embedding(codebook_size, output_steps * output_dim)
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_steps * output_dim),
        )
        self.output_steps = output_steps
        self.output_dim = output_dim

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        pooled = context.mean(dim=1)
        code_idx = self.logits(pooled).argmax(dim=-1)
        base = self.codebook(code_idx)
        residual = self.residual(pooled)
        return (base + residual).view(context.size(0), self.output_steps, self.output_dim)


def build_action_decoder(config: dict, hidden_dim: int, output_steps: int) -> nn.Module:
    decoder_type = config["type"]
    if decoder_type == "flow_matching":
        return FlowMatchingActionDecoder(hidden_dim=hidden_dim, output_steps=output_steps)
    if decoder_type == "mlp_codebook":
        codebook_size = int(config["mlp_codebook"]["codebook_size"])
        return MLPCodebookActionDecoder(hidden_dim=hidden_dim, output_steps=output_steps, codebook_size=codebook_size)
    if decoder_type == "consistency_distilled_diffusion":
        return FlowMatchingActionDecoder(hidden_dim=hidden_dim, output_steps=output_steps)
    raise ValueError(f"Unsupported action decoder type: {decoder_type}")
