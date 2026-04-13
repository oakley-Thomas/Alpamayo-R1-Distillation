"""Batch inference script for Alpamayo 1.5 teacher baseline.

Runs inference on all 100 clips from chunk 0, saves per-clip predictions
and ground truth, and reports aggregate minADE and RMSE.

Usage:
    python batch_inference.py [--data-dir /path/to/data] [--results-dir /path/to/results]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import physical_ai_av
import torch

from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

MODEL_ID = "nvidia/Alpamayo-1.5-10B"
# 6 samples gives a reliable minADE estimate; lower to 1 if VRAM is tight
NUM_TRAJ_SAMPLES = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alpamayo 1.5 batch inference baseline")
    parser.add_argument("--data-dir", type=Path, default=Path("/workspace/data"))
    parser.add_argument("--results-dir", type=Path, default=Path("/workspace/results"))
    parser.add_argument(
        "--chunk", type=int, default=0, help="Dataset chunk to evaluate (default: 0)"
    )
    parser.add_argument(
        "--num-traj-samples",
        type=int,
        default=NUM_TRAJ_SAMPLES,
        help="Trajectory samples per clip — more samples = better minADE, more VRAM",
    )
    parser.add_argument(
        "--t0-us",
        type=int,
        default=5_100_000,
        help="Timestamp offset in microseconds for each clip (default: 5.1s)",
    )
    parser.add_argument(
        "--clip-offset",
        type=int,
        default=0,
        help="Skip the first N clips (for splitting work across GPUs)",
    )
    parser.add_argument(
        "--clip-limit",
        type=int,
        default=None,
        help="Maximum number of clips to evaluate (for splitting work across GPUs)",
    )
    return parser.parse_args()


def compute_metrics(
    pred_xyz: torch.Tensor, gt_xyz: np.ndarray
) -> tuple[float, float, int]:
    """Compute minADE and RMSE over trajectory samples.

    Args:
        pred_xyz: Model output, shape (1, 1, num_samples, T, 3).
        gt_xyz: Ground truth future positions, shape (T, 3).

    Returns:
        min_ade: Minimum ADE across trajectory samples (metres).
        rmse: RMSE of the best sample vs ground truth (metres).
        best_sample_idx: Index of the trajectory sample with lowest ADE.
    """
    gt_xy = gt_xyz[:, :2]  # (T, 2) — evaluate in the XY plane

    # pred: (num_samples, T, 2)
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2]

    # ADE per sample: mean L2 distance over time
    diff = np.linalg.norm(pred_xy - gt_xy[None], axis=-1)  # (num_samples, T)
    ade_per_sample = diff.mean(axis=-1)  # (num_samples,)
    best_idx = int(ade_per_sample.argmin())
    min_ade = float(ade_per_sample[best_idx])

    # RMSE of the best sample
    best_pred = pred_xy[best_idx]  # (T, 2)
    rmse = float(np.sqrt(((best_pred - gt_xy) ** 2).sum(axis=-1).mean()))

    return min_ade, rmse, best_idx


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load clip IDs from the downloaded index                             #
    # ------------------------------------------------------------------ #
    clip_index_path = args.data_dir / "clip_index.parquet"
    if not clip_index_path.exists():
        sys.exit(f"clip_index.parquet not found at {clip_index_path}. Run download_data.py first.")

    clip_index = pd.read_parquet(clip_index_path)
    clip_ids = clip_index[
        (clip_index["chunk"] == args.chunk) & (clip_index["clip_is_valid"] == True)
    ].index.tolist()

    clip_ids = clip_ids[args.clip_offset:]
    if args.clip_limit is not None:
        clip_ids = clip_ids[: args.clip_limit]

    print(f"Clips to evaluate: {len(clip_ids)} (chunk {args.chunk}, offset {args.clip_offset})")

    # ------------------------------------------------------------------ #
    # Load model (once)                                                   #
    # ------------------------------------------------------------------ #
    print(f"\nLoading {MODEL_ID} ...")
    model = Alpamayo1_5.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda")
    model.eval()
    processor = helper.get_processor(model.tokenizer)
    print("Model ready.\n")

    # ------------------------------------------------------------------ #
    # Initialise dataset interface (once — avoids repeated HF auth/init)  #
    # ------------------------------------------------------------------ #
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    # ------------------------------------------------------------------ #
    # Inference loop                                                      #
    # ------------------------------------------------------------------ #
    results: list[dict] = []
    skipped: list[str] = []

    for i, clip_id in enumerate(clip_ids):
        prefix = f"[{i + 1:3d}/{len(clip_ids)}] {clip_id}"
        torch.cuda.reset_peak_memory_stats()
        t_start = time.perf_counter()

        # --- Load clip data -------------------------------------------
        try:
            data = load_physical_aiavdataset(
                clip_id,
                t0_us=args.t0_us,
                avdi=avdi,
                maybe_stream=True,
            )
        except Exception as exc:
            print(f"{prefix}  SKIP (load: {exc})")
            skipped.append(clip_id)
            continue

        # --- Build model inputs ----------------------------------------
        messages = helper.create_message(
            frames=data["image_frames"].flatten(0, 1),
            camera_indices=data["camera_indices"],
        )
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = helper.to_device(
            {
                "tokenized_data": inputs,
                "ego_history_xyz": data["ego_history_xyz"],
                "ego_history_rot": data["ego_history_rot"],
            },
            device="cuda",
        )

        # --- Run inference ---------------------------------------------
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, pred_rot, extra = (
                    model.sample_trajectories_from_data_with_vlm_rollout(
                        data=model_inputs,
                        top_p=0.98,
                        temperature=0.6,
                        num_traj_samples=args.num_traj_samples,
                        max_generation_length=256,
                        return_extra=True,
                    )
                )
        except Exception as exc:
            print(f"{prefix}  SKIP (inference: {exc})")
            skipped.append(clip_id)
            continue

        t_elapsed = time.perf_counter() - t_start
        vram_gb = torch.cuda.max_memory_allocated() / 1e9

        # --- Metrics ---------------------------------------------------
        gt_xyz = data["ego_future_xyz"].cpu().numpy()[0, 0]  # (T, 3)
        min_ade, rmse, best_idx = compute_metrics(pred_xyz, gt_xyz)

        print(
            f"{prefix}  "
            f"minADE={min_ade:.3f}m  RMSE={rmse:.3f}m  "
            f"t={t_elapsed:.1f}s  VRAM={vram_gb:.1f}GB"
        )

        results.append(
            {
                "clip_id": clip_id,
                "min_ade": min_ade,
                "rmse": rmse,
                "best_sample_idx": best_idx,
                "latency_s": round(t_elapsed, 2),
                "vram_peak_gb": round(vram_gb, 2),
                # (num_samples, T, 3) — keep all samples for later analysis
                "pred_xyz": pred_xyz.cpu().numpy()[0, 0].tolist(),
                "gt_xyz": gt_xyz.tolist(),
                "cot": extra["cot"][0, 0, 0] if "cot" in extra else "",
            }
        )

        # Save incrementally so a crash mid-run doesn't lose everything
        with open(args.results_dir / "results.json", "w") as f:
            json.dump(results, f)

    # ------------------------------------------------------------------ #
    # Summary                                                             #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("Alpamayo 1.5 — Teacher Baseline Results")
    print("=" * 60)

    if not results:
        print("No clips evaluated successfully.")
        return

    min_ades = [r["min_ade"] for r in results]
    rmses = [r["rmse"] for r in results]
    latencies = [r["latency_s"] for r in results]
    vrams = [r["vram_peak_gb"] for r in results]

    print(f"Clips evaluated : {len(results)} / {len(clip_ids)}")
    if skipped:
        print(f"Clips skipped   : {len(skipped)}")
    print(f"minADE          : {np.mean(min_ades):.3f}m  (std {np.std(min_ades):.3f}m)")
    print(f"RMSE            : {np.mean(rmses):.3f}m  (std {np.std(rmses):.3f}m)")
    print(f"Latency / clip  : {np.mean(latencies):.1f}s  (total {sum(latencies) / 60:.1f} min)")
    print(f"Peak VRAM       : {np.max(vrams):.1f}GB")

    summary = {
        "model": MODEL_ID,
        "chunk": args.chunk,
        "num_traj_samples": args.num_traj_samples,
        "n_clips_evaluated": len(results),
        "n_clips_skipped": len(skipped),
        "mean_min_ade": float(np.mean(min_ades)),
        "std_min_ade": float(np.std(min_ades)),
        "mean_rmse": float(np.mean(rmses)),
        "std_rmse": float(np.std(rmses)),
        "mean_latency_s": float(np.mean(latencies)),
        "total_latency_min": float(sum(latencies) / 60),
        "peak_vram_gb": float(np.max(vrams)),
    }

    summary_path = args.results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {args.results_dir}/")


if __name__ == "__main__":
    main()
