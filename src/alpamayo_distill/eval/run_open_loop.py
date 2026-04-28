from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from alpamayo_distill.eval.alignment import token_trajectory_alignment_score
from alpamayo_distill.eval.open_loop import compute_open_loop_metrics
from alpamayo_distill.training.common import load_training_bundle


def evaluate_model(max_batches: int | None = None) -> dict[str, float]:
    model, _cfg, dataloader = load_training_bundle()
    model.eval()
    aggregate = {
        "student_minADE": 0.0,
        "student_minFDE": 0.0,
        "student_RMSE": 0.0,
        "teacher_minADE": 0.0,
        "teacher_minFDE": 0.0,
        "teacher_RMSE": 0.0,
        "alignment_score": 0.0,
    }
    batches = 0
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(batch)
            student_metrics = compute_open_loop_metrics(outputs.trajectory, batch["groundtruth_trajectory"])
            teacher_metrics = compute_open_loop_metrics(batch["teacher_trajectory"], batch["groundtruth_trajectory"])
            aggregate["student_minADE"] += student_metrics["minADE"]
            aggregate["student_minFDE"] += student_metrics["minFDE"]
            aggregate["student_RMSE"] += student_metrics["RMSE"]
            aggregate["teacher_minADE"] += teacher_metrics["minADE"]
            aggregate["teacher_minFDE"] += teacher_metrics["minFDE"]
            aggregate["teacher_RMSE"] += teacher_metrics["RMSE"]
            aggregate["alignment_score"] += token_trajectory_alignment_score(outputs.cot_text, outputs.trajectory)
            batches += 1
            if max_batches is not None and batches >= max_batches:
                break
    if batches == 0:
        raise RuntimeError("No evaluation batches available")
    summary = {key: value / batches for key, value in aggregate.items()}
    summary["student_teacher_minADE_ratio"] = summary["teacher_minADE"] / max(summary["student_minADE"], 1e-6)
    summary["student_teacher_minFDE_ratio"] = summary["teacher_minFDE"] / max(summary["student_minFDE"], 1e-6)
    summary["student_teacher_RMSE_ratio"] = summary["teacher_RMSE"] / max(summary["student_RMSE"], 1e-6)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    summary = evaluate_model(max_batches=args.max_batches)
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
