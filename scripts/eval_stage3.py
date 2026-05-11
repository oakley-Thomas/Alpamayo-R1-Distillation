"""Evaluate Stage 3 flow-matching Action Expert acceptance metrics."""

from __future__ import annotations

import argparse
import json
import os

# CuBLAS reads this before torch is imported by the evaluation module.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from src.eval.stage3 import run_stage3_evaluation, write_stage3_eval_report


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run Stage 3 evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage3.yaml")
    parser.add_argument("--split", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--predictions-output", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--measure-latency", action="store_true")
    args = parser.parse_args(argv)

    report = run_stage3_evaluation(
        config_path=args.config,
        split_file=args.split,
        checkpoint_path=args.checkpoint,
        predictions_output=args.predictions_output,
        measure_latency=args.measure_latency,
    )
    if args.output is not None:
        write_stage3_eval_report(report, args.output)
    print(json.dumps(report.to_json_dict(), indent=2))
    return 0 if report.passes_acceptance else 1


if __name__ == "__main__":
    raise SystemExit(main())
