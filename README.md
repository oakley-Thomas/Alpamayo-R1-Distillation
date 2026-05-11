# Alpamayo-1.5 Distillation

Compressing NVIDIA's 10B-parameter Alpamayo-1.5 autonomous-driving Vision-Language-Action model into a ~5B student that fits on a single 24 GB GPU — without giving up the chain-of-causation reasoning or trajectory quality that make the teacher useful in the first place.

This README explains *what* we're doing and *why*. For implementation specifics, see [`DISTILLATION_PLAN.md`](./DISTILLATION_PLAN.md).

---

## What is Alpamayo-1.5?

Alpamayo-1.5 is NVIDIA's open-source Vision-Language-Action (VLA) model for autonomous driving. Unlike traditional self-driving stacks that map perception directly to control, Alpamayo introduces an explicit reasoning step in the middle:

```
Camera frames → Vision encoder → Chain-of-Causation reasoning → Action Expert → Trajectory
```

The model has two main components totaling ~10B parameters:

- **Cosmos Reasoning Backbone (8.2B)** — NVIDIA's world model, post-trained with RL to produce *chain-of-causation* reasoning traces. Instead of just predicting "turn right," it explains *why*: "Nudge to the right to avoid the construction cones encroaching on the lane."
- **Action Expert (2.3B)** — A diffusion-based trajectory generator that converts the reasoning into 64 waypoints over 6.4 seconds (10 Hz).

The reasoning isn't decorative. The Action Expert is conditioned on the VLM's hidden states — states that have been shaped by RL post-training — not on the raw text. So the chain-of-causation is both an explainability feature *and* a representation-shaping mechanism.

## Why distill it?

Inference VRAM scales sharply with the number of trajectory samples:

| Trajectory Samples | VRAM |
|---|---|
| 1 | 24 GB |
| 16 | 40 GB |
| 16 + Classifier-Free Guidance | 60 GB |

UB's Autonomous Lincoln runs an RTX 4090 with **24 GB**. That's exactly enough for a single trajectory sample with no headroom for anything else on the system — no perception, no logging, no margin. Production driving wants 16+ samples to pick from, plus classifier-free guidance to bias toward safer trajectories. Alpamayo-1.5 cannot deliver that on the hardware we actually have.

The goal is a student model that:

- Fits in **24 GB** with room for the rest of the stack.
- Runs in **under 200 ms** end-to-end, single-batch.
- Stays close to the teacher on **trajectory error** at all horizons.
- Preserves the **chain-of-causation reasoning** so the model remains explainable.

Target size: **~5B parameters**, half the original.

## Student architecture

| Component | Teacher | Student |
|---|---|---|
| VLM backbone | Cosmos Reason (8.2B) | Qwen2.5-VL-3B |
| Action Expert | Multi-step diffusion (2.3B) | Single-step flow matching (~0.3B) |
| Total | ~10B | ~5B (with QLoRA-quantized backbone) |

Two architectural shifts do the heavy lifting on size and speed:

1. **Smaller VLM backbone.** Qwen2.5-VL-3B is a well-supported open VLM that's small enough to QLoRA-quantize to 4-bit and run alongside the rest of the system.
2. **Diffusion → single-step flow matching.** The teacher's diffusion Action Expert runs many denoising steps per trajectory. Flow matching reformulates the same problem as a velocity field that, once trained, can be integrated in a single Euler step. One forward pass instead of dozens.

## The distillation approach

This is where the project gets interesting. Naive distillation — train the student to copy the teacher's outputs — would fail here, because Alpamayo's value lives in the *internal* representations the RL post-training carved out, not in the text it emits. So the pipeline matches the teacher at multiple levels.

### Stage 1: Teacher data generation *(complete)*

Run teacher inference on 100 driving clips from NVIDIA's open Physical AI dataset and dump everything we'll need downstream:

- The chain-of-causation reasoning trace.
- The hidden states that feed the Action Expert.
- The full denoising trajectory of the diffusion model.
- 16 sampled 64-waypoint trajectories per clip.

This took ~12 hours on an RTX Quadro 8000 (48 GB) and is already done. Stages 2–4 read from this dump rather than re-running the teacher.

### Stage 2: VLM backbone distillation

Train Qwen2.5-VL-3B to imitate Cosmos Reason on **two levels simultaneously**:

```
L_stage2 = L_CoC-KL + α · L_align + β · L_LM-CE
```

- **`L_CoC-KL`** — KL divergence on the reasoning-trace token distribution. Teaches the student to *speak* like the teacher.
- **`L_align`** — Smooth-L1 between the student's hidden states (after a learned adapter) and the teacher's hidden states at the same positions. Teaches the student to *represent* the scene like the teacher.
- **`L_LM-CE`** — Plain cross-entropy on teacher tokens. Acts as a regularizer when the soft KL signal is uninformative.

The hidden-state alignment is the load-bearing piece. Without it, the student would learn to produce fluent reasoning text while feeding the Action Expert a representation that means something different than what the teacher's downstream module expects. The text would look right and the trajectories would be wrong.

### Stage 3: Action Expert distillation

With the Stage 2 backbone frozen, train a new flow-matching Action Expert that produces single-step trajectories from the student's hidden states.

```
L_stage3 = L_FM + γ · L_traj
```

- **`L_FM`** — Standard rectified-flow loss: learn the velocity field that transports Gaussian noise to teacher-sampled trajectories.
- **`L_traj`** — Direct regression of the *single-step* student prediction onto the teacher trajectory. This matters because flow-matching theory guarantees correctness in the many-step limit; what we deploy is the one-step approximation, which needs its own supervision to stay accurate.

Trajectories are represented as `(x, y, heading)` in ego frame, normalized per-clip.

### Stage 4: End-to-end alignment fine-tuning

Unfreeze both modules and train end-to-end with one new term added to the Stage 2 + Stage 3 losses:

```
L_stage4 = L_stage2 + L_stage3 + δ · L_consistency
```

The consistency loss penalizes cases where the student's *stated reasoning* and *emitted trajectory* disagree. The slide deck calls this "approximating RL," which is honest framing — there's no policy gradient or reward model. The mechanism is supervised: extract a canonical action label from the student's CoC trace (e.g., "yield," "nudge-right," "follow-lead"), look up the teacher's prototype trajectory for that action, and regress the student's trajectory toward it when self-generated reasoning and teacher reasoning diverge.

This stage is opt-in. If it doesn't improve evaluation metrics, we ship the Stage 3 checkpoint and call it done.

## Why this stage ordering

Each stage builds the inputs the next stage needs and freezes them:

- Stage 2 produces hidden states. The Action Expert in Stage 3 needs them as conditioning.
- Stage 3 produces a working trajectory head. Stage 4 needs both modules trainable but starting from a good initialization, otherwise the consistency loss has no signal to align.
- Going straight to end-to-end training would mean every loss term is fighting every other one in a half-trained network. Staged training lets each component reach a working state on its own clean objective before the joint objective brings them together.

## Evaluation

The student is benchmarked on a 10-clip held-out subset of the NVIDIA Physical AI dataset, head-to-head with the teacher, on:

- **Trajectory error** — ADE and FDE at 1s / 3s / 6s horizons, against the teacher trajectory.
- **Latency** — end-to-end on RTX 4090, batch 1, median of 100 trials.
- **Memory footprint** — peak VRAM during inference.
- **Parameter count** — total and trainable.

Project success criteria:

| Metric | Target |
|---|---|
| Total parameters | ≤ 5.5 B |
| Peak inference VRAM | ≤ 24 GB |
| ADE @ 6s vs. teacher | ≤ 0.7 m |
| End-to-end latency | ≤ 200 ms |

## Hardware

- **Training:** Single RTX Quadro 8000, 48 GB. QLoRA on the backbone is required to fit Stage 4.
- **Deployment target:** RTX 4090, 24 GB (UB's Autonomous Lincoln).

## What this project is *not*

- Not a deployment to a real vehicle. Integration with the Autonomous Lincoln, ROS, sensor fusion, and safety certification are all downstream of this work.
- Not real reinforcement learning. The Stage 4 "alignment" loss is a supervised proxy, not policy optimization. Real RL distillation (e.g., DPO over trajectory pairs) is future work.
- Not a new VLA architecture. We're compressing an existing one.

## Repository

```
alpamayo-distill/
├── README.md                   # this file
├── DISTILLATION_PLAN.md        # detailed engineering handoff
├── configs/                    # one YAML per training run
├── data/
│   ├── teacher_dump/           # Stage 1 outputs (read-only for Stages 2–4)
│   └── splits/                 # train / val / test clip IDs
├── src/
│   ├── data/                   # loaders, validators
│   ├── models/                 # student VLM, Action Expert, teacher interface
│   ├── losses/                 # CoC-KL, hidden-state align, flow-matching, consistency
│   ├── train/                  # one trainer per stage
│   └── eval/                   # metrics, latency, memory profiling
├── scripts/                    # CLI entry points
└── tests/                      # required per stage
```

## Authors

Oakley Thomas and Harsh Shinde — University at Buffalo.

## Acknowledgments

Built on NVIDIA's open release of Alpamayo-1.5 and the Physical AI Autonomous Vehicles dataset. Backbone is Qwen2.5-VL-3B from Alibaba.
