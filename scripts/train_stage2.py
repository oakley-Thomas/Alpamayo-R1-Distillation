"""Run Stage 2 VLM backbone distillation."""

from __future__ import annotations

import argparse

from src.train.stage2 import run_stage2_training


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run Stage 2 training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2.yaml")
    args = parser.parse_args(argv)
    run_stage2_training(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
