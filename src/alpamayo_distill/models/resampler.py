from __future__ import annotations

import torch
from torch import nn


class IdentityResampler(nn.Module):
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens
