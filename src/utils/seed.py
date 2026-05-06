"""Deterministic seed setup for training scripts."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set Python, NumPy, PyTorch, and CUDA seeds.

    Args:
        seed: Seed value used for all RNGs.
        deterministic: Enable deterministic PyTorch kernels where feasible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
