from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from alpamayo_distill.training.common import load_training_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/example_prediction.json")
    args = parser.parse_args()
    model, _cfg, dataloader = load_training_bundle()
    batch = next(iter(dataloader))
    model.eval()
    with torch.no_grad():
        outputs = model(batch)
    payload = {
        "trajectory": outputs.trajectory[0].cpu().tolist(),
        "cot_text": outputs.cot_text[0],
        "teacher_trajectory": batch["teacher_trajectory"][0].cpu().tolist(),
        "groundtruth_trajectory": batch["groundtruth_trajectory"][0].cpu().tolist(),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote prediction example to {out_path}")


if __name__ == "__main__":
    main()
