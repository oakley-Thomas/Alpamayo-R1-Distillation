# Alpamayo Student V1

## Summary

This artifact is the v1 student scaffold for Alpamayo teacher-student distillation. The current repository includes:

- Config-driven training stages for warmup, action-head training, and joint fine-tuning.
- Offline teacher rollout caching with top-k logit compression.
- Open-loop evaluation, latency benchmarking, and example inference output.

## Intended Use

- Research on open-loop autonomous-driving trajectory distillation.
- Controlled benchmarking of student-vs-teacher tradeoffs in memory, latency, and trajectory quality.

## Current Limitations

- The default code path uses a local stub backbone and stub teacher rollout for offline verification.
- Closed-loop AlpaSim evaluation is not implemented in v1.
- RL post-training is intentionally disabled in v1.

## Training Data Contract

Each rolled-out scene is expected to contain:

- `cameras`
- `egomotion`
- `teacher_tokens`
- `teacher_topk_logits`
- `teacher_topk_indices`
- `teacher_trajectory`
- `groundtruth_trajectory`

## Output Artifacts

- `ckpts/stage1_warmup.pt`
- `ckpts/stage2_action.pt`
- `ckpts/stage3_joint.pt`
- `artifacts/example_prediction.json`
