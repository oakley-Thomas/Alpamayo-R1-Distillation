"""Tests for Stage 3 trajectory metrics."""

from __future__ import annotations

import math

import torch

from src.eval.stage3 import (
    evaluate_stage3_predictions,
    heading_continuity_rate,
    trajectory_ade,
    trajectory_fde,
)


def test_trajectory_ade_and_fde_use_xy_displacement() -> None:
    predictions = torch.zeros((2, 64, 3))
    targets = torch.zeros((2, 64, 3))
    targets[..., 0] = 3.0
    targets[..., 1] = 4.0
    targets[..., 2] = 100.0

    assert trajectory_ade(predictions, targets).item() == 5.0
    assert trajectory_fde(predictions, targets).item() == 5.0


def test_heading_continuity_rate_detects_large_jumps() -> None:
    predictions = torch.zeros((2, 64, 3))
    predictions[0, :, 2] = 0.1
    predictions[1, 1, 2] = math.pi

    assert heading_continuity_rate(predictions).item() == 0.5


def test_evaluate_stage3_predictions_reports_passes() -> None:
    predictions = torch.zeros((2, 64, 3))
    targets = torch.zeros((2, 64, 3))

    report = evaluate_stage3_predictions(
        predictions=predictions,
        targets=targets,
        split_file="val.json",
        latency_ms=10.0,
    )

    assert report.ade_m == 0.0
    assert report.fde_m == 0.0
    assert report.heading_continuity_rate == 1.0
    assert report.passes_acceptance
