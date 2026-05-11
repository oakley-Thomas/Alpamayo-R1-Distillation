"""Deterministic seed setup for training scripts."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any, cast

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
    manual_seed = cast(Callable[[int], Any], object.__getattribute__(torch, "manual_seed"))
    manual_seed_all = cast(
        Callable[[int], Any], object.__getattribute__(torch.cuda, "manual_seed_all")
    )
    manual_seed(seed)
    manual_seed_all(seed)
    if deterministic:
        deterministic_algorithms = cast(
            Callable[..., Any], object.__getattribute__(torch, "use_deterministic_algorithms")
        )
        deterministic_algorithms(True, warn_only=True)
