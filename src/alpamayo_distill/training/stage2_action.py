from __future__ import annotations

from alpamayo_distill.training.common import load_training_bundle, run_single_epoch, save_stage_artifacts


def main() -> None:
    model, cfg, dataloader = load_training_bundle()
    metrics = run_single_epoch(model, cfg, dataloader, lr=float(cfg["stages"]["stage2_action"]["lr"]))
    save_stage_artifacts(model, "stage2_action", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
