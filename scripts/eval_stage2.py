"""Evaluate Stage 2 VLM backbone acceptance metrics."""

from __future__ import annotations

import argparse
import json

from src.eval.stage2 import run_stage2_evaluation, write_stage2_eval_report


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run Stage 2 evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2.yaml")
    parser.add_argument("--split", default=None)
    parser.add_argument("--student-dir", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    report = run_stage2_evaluation(
        config_path=args.config,
        split_file=args.split,
        student_dir=args.student_dir,
    )
    if args.output is not None:
        write_stage2_eval_report(report, args.output)
    print(json.dumps(report.to_json_dict(), indent=2))
    return 0 if report.passes_acceptance else 1


if __name__ == "__main__":
    raise SystemExit(main())
