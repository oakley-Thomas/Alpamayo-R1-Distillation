from __future__ import annotations

from alpamayo_distill.training.common import load_training_bundle, run_single_epoch


def main() -> None:
    model, cfg, dataloader = load_training_bundle()
    metrics = run_single_epoch(model, cfg, dataloader, lr=float(cfg["stages"]["stage1_warmup"]["lr"]))
    print(metrics)


if __name__ == "__main__":
    main()
