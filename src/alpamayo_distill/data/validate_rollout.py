from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from alpamayo_distill.utils.logits_storage import LogitsStorage, dequantize_topk_logits


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


def validate_scene(payload: dict, scale: float) -> dict[str, float]:
    topk_logits = dequantize_topk_logits(np.asarray(payload["teacher_topk_logits"]), scale=scale)
    probs = softmax(topk_logits)
    mass = probs.sum(axis=-1).mean()
    teacher_traj = np.asarray(payload["teacher_trajectory"])
    gt_traj = np.asarray(payload["groundtruth_trajectory"])
    traj_rmse = float(np.sqrt(((teacher_traj - gt_traj) ** 2).mean()))
    return {
        "token_count": float(np.asarray(payload["teacher_tokens"]).shape[0]),
        "topk_probability_mass": float(mass),
        "trajectory_rmse": traj_rmse,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="artifacts/teacher_rollouts")
    parser.add_argument("--max-scenes", type=int, default=20)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    files = sorted(cache_dir.glob("*.h5")) + sorted(cache_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"No rollout files found in {cache_dir}")

    storage = LogitsStorage(cache_dir)
    metrics = []
    total_bytes = 0
    for path in files[: args.max_scenes]:
        total_bytes += path.stat().st_size
        payload = storage.read_scene(path)
        metrics.append(validate_scene(payload, scale=float(payload.get("logit_scale", storage.scale))))

    avg_mass = sum(item["topk_probability_mass"] for item in metrics) / len(metrics)
    avg_tokens = sum(item["token_count"] for item in metrics) / len(metrics)
    avg_rmse = sum(item["trajectory_rmse"] for item in metrics) / len(metrics)
    avg_bytes = total_bytes / len(metrics)
    print(
        {
            "scenes_checked": len(metrics),
            "avg_token_count": round(avg_tokens, 2),
            "avg_topk_probability_mass": round(avg_mass, 6),
            "avg_teacher_gt_rmse": round(avg_rmse, 6),
            "avg_scene_bytes": round(avg_bytes, 2),
        }
    )


if __name__ == "__main__":
    main()
