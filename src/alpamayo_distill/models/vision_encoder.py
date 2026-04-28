from __future__ import annotations

import torch
from torch import nn


class SimpleVisionEncoder(nn.Module):
    def __init__(self, output_dim: int, tokens_per_camera: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.tokens_per_camera = tokens_per_camera
        self.proj = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=4, padding=3),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((tokens_per_camera, 1)),
        )
        self.head = nn.Linear(64, output_dim)

    def forward(self, cameras: torch.Tensor) -> torch.Tensor:
        batch, num_cams, channels, height, width = cameras.shape
        flat = cameras.view(batch * num_cams, channels, height, width)
        feats = self.proj(flat).squeeze(-1).transpose(1, 2)
        feats = self.head(feats)
        return feats.view(batch, num_cams * self.tokens_per_camera, self.output_dim)
