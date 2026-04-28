from __future__ import annotations

from typing import Iterable

import torch


def pad_token_sequences(sequences: Iterable[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
    seqs = list(sequences)
    if not seqs:
        return torch.empty(0, 0, dtype=torch.long)
    max_len = max(seq.numel() for seq in seqs)
    output = torch.full((len(seqs), max_len), pad_value, dtype=seqs[0].dtype)
    for row, seq in enumerate(seqs):
        output[row, : seq.numel()] = seq
    return output


def make_attention_mask(tokens: torch.Tensor, pad_value: int = 0) -> torch.Tensor:
    return (tokens != pad_value).long()
