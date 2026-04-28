from __future__ import annotations

import argparse

from alpamayo_distill.eval.latency import benchmark_latency
from alpamayo_distill.training.common import load_training_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    model, _cfg, dataloader = load_training_bundle()
    batch = next(iter(dataloader))
    print(benchmark_latency(model, batch, iterations=args.iterations, warmup_iterations=args.warmup))


if __name__ == "__main__":
    main()
