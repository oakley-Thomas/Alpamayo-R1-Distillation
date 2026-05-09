"""Export Alpamayo teacher outputs to the Stage 1 dump contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.export_teacher_dump import ExportTeacherDumpConfig, export_teacher_dump


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and export the teacher dump."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-ids", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-name", default="nvidia/Alpamayo-1.5-10B")
    parser.add_argument("--num-traj-samples", type=int, default=16)
    parser.add_argument("--traj-sample-batch-size", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--max-generation-length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--t0-us", type=int, default=5_100_000)
    parser.add_argument("--include-kv-cache", action="store_true")
    parser.add_argument("--capture-denoising", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--stage2-only",
        action="store_true",
        help="Allow export without Stage 3 denoising fields.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    export_teacher_dump(
        ExportTeacherDumpConfig(
            clip_ids=Path(args.clip_ids),
            output_root=Path(args.output_root),
            model_name=args.model_name,
            num_traj_samples=args.num_traj_samples,
            traj_sample_batch_size=args.traj_sample_batch_size,
            top_k=args.top_k,
            max_generation_length=args.max_generation_length,
            temperature=args.temperature,
            top_p=args.top_p,
            t0_us=args.t0_us,
            include_kv_cache=args.include_kv_cache,
            capture_denoising=args.capture_denoising,
            require_stage3_fields=not args.stage2_only,
            overwrite=args.overwrite,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
