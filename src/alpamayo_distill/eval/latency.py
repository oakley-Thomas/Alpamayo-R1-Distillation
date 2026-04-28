from __future__ import annotations

import time
from statistics import mean

import torch


def benchmark_latency(model, batch: dict[str, torch.Tensor], iterations: int = 10) -> dict[str, float]:
    samples = []
    model.eval()
    with torch.no_grad():
        for _ in range(iterations):
            start = time.perf_counter()
            model(batch)
            samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return {
        "p50_ms": samples[len(samples) // 2],
        "p95_ms": samples[min(len(samples) - 1, int(len(samples) * 0.95))],
        "mean_ms": mean(samples),
    }
