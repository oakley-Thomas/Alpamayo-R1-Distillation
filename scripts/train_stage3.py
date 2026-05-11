"""Run Stage 3 flow-matching Action Expert distillation."""

from __future__ import annotations

import argparse
import os

# CuBLAS reads this before torch is imported by the training module.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from src.train.stage3 import run_stage3_training


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run Stage 3 training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage3.yaml")
    args = parser.parse_args(argv)
    run_stage3_training(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
