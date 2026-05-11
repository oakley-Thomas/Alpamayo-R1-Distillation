# Alpamayo-1.5 Distillation: Engineering Handoff Plan

**Project:** Compress Alpamayo-1.5 (~10B params: 8.2B Cosmos-Reason VLM + 2.3B diffusion Action Expert) into a ~5B student (Qwen2.5-VL-3B + single-step flow-matching Action Expert) that fits in 24 GB VRAM at inference and matches teacher trajectory quality on the NVIDIA Physical AI held-out set.

**Audience:** LLM coding agents implementing the pipeline. Each section is written so a single agent can be assigned a stage with clear inputs, outputs, and acceptance criteria.

**Hardware target:** Single GPU, 24–48 GB VRAM (RTX Quadro 8000 48 GB available for training; RTX 4090 24 GB is the deployment target).

**Status assumed at start of work:** Stage 1 complete. Teacher inference outputs already on disk (CoC traces, hidden states, denoising trajectories, 64-waypoint trajectories) over 100 driving clips × 16 trajectories each.

---

## 0. Conventions and Contracts

These apply to every stage. Agents must not deviate without raising a question.

**Repository layout (target):**

```
alpamayo-distill/
├── configs/                  # YAML per stage; one file per run
├── data/
│   ├── teacher_dump/         # Stage 1 outputs (READ ONLY for Stages 2–4)
│   └── splits/               # train.json, val.json, test.json (clip IDs)
├── src/
│   ├── data/                 # Dataset, collate, streaming loaders
│   ├── models/
│   │   ├── student_vlm.py    # Qwen2.5-VL-3B + adapter heads
│   │   ├── action_expert.py  # Flow-matching head
│   │   └── teacher_iface.py  # Wrapper: load Stage 1 dump, no live inference
│   ├── losses/               # CoC-KL, hidden-state align, flow-matching, alignment
│   ├── train/                # One trainer per stage: stage2.py, stage3.py, stage4.py
│   └── eval/                 # Trajectory metrics, latency, memory profiler
├── scripts/                  # CLI entry points
└── tests/                    # Unit tests; required per stage (see acceptance criteria)
```

**Frameworks:** PyTorch 2.4+, Hugging Face Transformers, `peft` for LoRA/QLoRA, `bitsandbytes` for 4-bit quantization, `accelerate` for mixed precision and gradient checkpointing. Do not introduce DeepSpeed or FSDP — single-GPU only.

**Precision policy:** Train in bf16. Quantize the frozen teacher hidden-state cache to fp16 on disk to halve I/O. The Qwen2.5-VL-3B backbone is loaded in 4-bit NF4 with QLoRA adapters (this is the only way to fit Stage 4 on a single 48 GB card).

**Reproducibility:** Every run writes `run_config.yaml`, `git_sha.txt`, and `env.txt` (pip freeze) to its output directory. Seeds set for `torch`, `numpy`, `random`, and `torch.cuda`.

**Logging:** Weights & Biases if available, else TensorBoard. Log every loss component separately — never log only the combined loss.

**Coding-agent guardrails:** No silent fallback to CPU. No swallowed exceptions in data loading (a corrupt teacher dump entry must fail loudly with the offending clip ID). No live teacher inference outside Stage 1 — Stages 2–4 read from `data/teacher_dump/` only.

---

## 1. Data Contract: Stage 1 Teacher Dump

This is the input to everything downstream. Agents working on Stage 2+ must implement against this contract; if the on-disk format differs, fix the loader, do not rewrite the teacher.

**On-disk layout (one directory per clip):**

```
data/teacher_dump/<clip_id>/
├── meta.json                        # clip_id, num_frames, fps, camera_intrinsics, source_url
├── frames/                          # multi-view RGB, one .jpg per (frame_idx, view)
│   └── frame_{NNNN}_{view}.jpg
├── coc_trace.json                   # list of {step_idx, text, token_ids, top_k_logits}
├── hidden_states.npy                # shape (T, D_h); D_h = teacher VLM hidden dim feeding Action Expert
├── denoising_traj.npz               # arrays: x_t (S, 64, 3), t (S,), v_pred (S, 64, 3); S = diffusion steps
└── trajectories.npy                 # shape (16, 64, 3); 16 sampled trajectories, 64 waypoints, (x, y, heading)
```

`top_k_logits` stores the top-32 logits per generated token (sufficient for KL on a soft-label distribution without storing the full 150k-vocab tensor — about 10 MB per clip vs. 50 GB).

**Per-stage view of this data:**

| Stage | Reads |
|-------|-------|
| 2 | `frames/`, `coc_trace.json` (token_ids + top_k_logits), `hidden_states.npy` |
| 3 | `hidden_states.npy` (student's, computed from Stage 2 model), `denoising_traj.npz`, `trajectories.npy` |
| 4 | All of the above |
| Eval | `trajectories.npy` as ground truth proxy |

**Splits:** 100 clips → 80 train / 10 val / 10 test, stratified by ego-action category if metadata permits, else random with fixed seed 42. Splits are written once to `data/splits/` and never regenerated.

---

## 2. Stage 2: VLM Backbone Distillation

**Goal:** Make student Qwen2.5-VL-3B produce (a) CoC reasoning traces close to the teacher's token distribution, and (b) Action-Expert-feeding hidden states close to the teacher's hidden states. The second is the load-bearing requirement; CoC text is a means to that end.

**Why both losses:** Slide 11 highlights that the teacher's Action Expert is conditioned on hidden states shaped by RL post-training, not on the raw CoC text. Matching tokens alone leaves the hidden-state geometry unconstrained, so the student would speak fluently and act poorly.

### 2.1 Model

- **Backbone:** Qwen2.5-VL-3B-Instruct, loaded 4-bit NF4 via `bitsandbytes`. Apply QLoRA to attention `q_proj`, `k_proj`, `v_proj`, `o_proj` and MLP `gate_proj`, `up_proj`, `down_proj`. Rank 64, alpha 128, dropout 0.05. The vision encoder stays frozen and is **not** wrapped with LoRA.
- **Hidden-state adapter:** `student_hidden = LayerNorm(Linear(d_student → d_teacher))`. The teacher's `D_h` (read from a Stage 1 sample at startup) defines the output dim. This adapter is full-precision and trainable. Hidden states for alignment are taken from the last decoder layer at the position(s) that condition the teacher's Action Expert — replicate the teacher's slicing logic; document it in `teacher_iface.py`.

### 2.2 Loss

The slide gives this directly:

```
L_stage2 = L_CoC-KL + α · L_align + β · L_LM-CE
```

Component definitions agents must implement:

- **`L_CoC-KL`** — KL divergence between student's next-token distribution and the teacher's top-32 logits (renormalized over the top-32 set), summed over CoC trace tokens, mean-reduced over batch. Use temperature 2.0 on both sides.
- **`L_align`** — Smooth-L1 between `student_hidden` and the teacher hidden states, per timestep, masked by valid-frame mask. Smooth-L1 (not MSE) because hidden-state magnitudes have outliers at scene transitions that MSE over-weights.
- **`L_LM-CE`** — Standard cross-entropy on teacher token IDs as hard labels. Acts as a regularizer when the top-32 KL is uninformative (e.g., low-entropy steps).

**Defaults:** α = 1.0, β = 0.1. Both must be configurable. Agent should run a 200-step ablation at α ∈ {0.3, 1.0, 3.0} on the val set and report which works best before the full run.

### 2.3 Training loop

- Optimizer: `paged_adamw_8bit` (bitsandbytes), lr 2e-4 for LoRA params, lr 1e-4 for the hidden-state adapter, weight decay 0.01, cosine schedule with 5% warmup.
- Effective batch: 1 clip per step, gradient accumulation 16. Each clip yields ~T frames; accumulate over frames inside the clip.
- Gradient checkpointing **on**.
- Memory budget at this stage: ~36 GB. If OOM, reduce LoRA rank to 32 first, then drop image resolution by 25%.
- Train for 3 epochs over the 80-clip train set. With ~T=200 steps/clip and 16-step accumulation, ~3000 optimizer steps total.

### 2.4 Outputs

- `outputs/stage2/student_vlm/` — LoRA adapter + hidden-state adapter weights (bundled).
- `outputs/stage2/student_hidden_cache/<clip_id>.npy` — student hidden states for all train+val+test clips. Stage 3 consumes this; do not skip the cache step.

### 2.5 Acceptance criteria

| Criterion | Target |
|-----------|--------|
| CoC token top-1 agreement with teacher (val) | ≥ 70% |
| Hidden-state cosine similarity (val, mean over timesteps) | ≥ 0.85 |
| Generated CoC trace is parseable (no truncation, no malformed JSON if structured) | 100% of val clips |
| Unit test: loader rejects clip with missing `hidden_states.npy` | passes |
| Unit test: KL is finite when teacher top-1 mass > 0.99 | passes |

If hidden-state cosine < 0.85, the agent must not proceed to Stage 3 — raise this and stop. Stage 3 cannot recover from a misaligned conditioning signal.

---

## 3. Stage 3: Action Expert Distillation

**Goal:** Replace the multi-step diffusion Action Expert with a single-step flow-matching head that, given the student's frozen Stage 2 hidden states, produces 64-waypoint trajectories matching the teacher's sampled trajectories.

**Why flow matching, not diffusion-distillation tricks like consistency models:** flow matching admits a clean single-step inference (one forward pass from noise to trajectory) and does not require teacher-step pairing during training. It also halves the parameter count (no separate noise predictor + scheduler state).

### 3.1 Model

- **Action Expert (student):** Transformer decoder, 8 layers, hidden dim 768, 12 heads, ~110 M params. Inputs: `(noise_sample ∈ R^{64×3}, t ∈ [0,1], hidden_state ∈ R^{T_cond × D_h})`. Output: `v_θ(x_t, t, h)`, the predicted velocity field.
- **Conditioning:** Cross-attention from waypoint queries to the student VLM's last-layer hidden states. The hidden states come from the cache built in §2.4 — Stage 2 is **frozen** during Stage 3.
- **Output head:** Linear(768 → 3) per waypoint position.

The student Action Expert is ~0.3 B params after this resizing, vs. the teacher's 2.3 B. Combined with the 3 B QLoRA-quantized backbone (~2 GB on disk, ~6 GB live), total student footprint comes in under 10 GB at inference — well inside 24 GB with headroom for activations and a batch of 1.

### 3.2 Loss

Standard rectified-flow objective, plus a teacher-trajectory regression to keep the student close to the teacher's sampled outputs (not just to ground truth):

```
L_stage3 = L_FM + γ · L_traj
```

- **`L_FM`** — Flow matching: sample `x_0 ~ N(0, I)`, take `x_1` = a teacher-sampled trajectory, set `x_t = (1-t) · x_0 + t · x_1` for `t ~ U(0,1)`, target velocity `u = x_1 - x_0`. Loss is `MSE(v_θ(x_t, t, h), u)`.
- **`L_traj`** — Smooth-L1 between the single-step student prediction `x̂_1 = x_0 + v_θ(x_0, 0, h)` and the teacher trajectory `x_1`. Forces the single-step inference path (which is what gets deployed) to match, not just the velocity field on average.

Default γ = 0.5.

**Trajectory representation:** `(x, y, heading)` in ego frame, normalized per-clip to mean 0, std 1, with stats stored alongside the checkpoint for de-normalization at inference.

### 3.3 Training loop

- Optimizer: AdamW, lr 3e-4, cosine schedule, 5% warmup, weight decay 0.01.
- Batch: 16 trajectories per step (each clip provides 16 teacher-sampled trajectories — use them all; do not collapse to a single mean trajectory).
- Stage 2 model in eval mode, hidden-state cache memory-mapped, no gradients flowing back. If hidden-state cache is missing, fail loudly.
- Train for 30 epochs over the 80-clip train set (Action Expert trains fast because the backbone is frozen).

### 3.4 Outputs

- `outputs/stage3/action_expert.pt` — checkpoint.
- `outputs/stage3/traj_norm_stats.json` — per-axis mean/std for de-normalization.
- `outputs/stage3/val_predictions.npz` — single-step predictions on val set, for §3.5 and human inspection.

### 3.5 Acceptance criteria

| Criterion | Target |
|-----------|--------|
| L2 ADE (Avg Displacement Error) vs. teacher trajectory, val | ≤ 0.5 m |
| L2 FDE (Final Displacement Error) vs. teacher trajectory, val | ≤ 1.5 m |
| Single-step inference latency on RTX 4090 | ≤ 50 ms (Action Expert only, batch 1) |
| Sanity: predicted heading is continuous (no >π/4 jumps between adjacent waypoints) on ≥ 99% of val trajectories | passes |

The third criterion is the project's core value proposition. If the student needs more than one step at inference to hit ADE/FDE targets, the experiment has failed and we should reconsider; raise this rather than silently increasing inference steps.

---

## 4. Stage 4: End-to-End Alignment Fine-Tuning

**Goal:** Unfreeze both modules and jointly fine-tune so the student's *stated reasoning* and *emitted trajectory* are mutually consistent and both match the teacher.

The slide is honest that this approximates RL — there is no policy gradient or reward model. The mechanism is an alignment loss that punishes reasoning/trajectory mismatch using the teacher's pairings as the gold standard.

### 4.1 Model

Both Stage 2 (Qwen2.5-VL-3B + LoRA + hidden-state adapter) and Stage 3 (Action Expert) are now trainable. LoRA stays — full fine-tuning of the 3B backbone will not fit on 48 GB once the Action Expert and activations are loaded.

### 4.2 Loss

```
L_stage4 = L_stage2 + L_stage3 + δ · L_consistency
```

`L_consistency` is the alignment term. Implementation:

1. Generate the student's CoC trace `c_s` (greedy or low-temp sample) for the input frames.
2. Compute the student's hidden states `h_s` from this self-generated trace (not from teacher-forced tokens).
3. Run the Action Expert to get a trajectory `x_s = action_expert(h_s)`.
4. Compute the teacher's hidden states `h_t` (from cache, conditioned on the teacher's CoC trace) and the teacher's trajectory `x_t`.
5. **Consistency loss** = `MSE(x_s, x_t) when CoC(c_s) ≈ CoC(c_t), penalty otherwise`. Concretely, score `c_s` against `c_t` with a CoC-similarity function (token-level F1 on a canonical action vocabulary — "yield", "merge", "nudge-right", etc., extracted via regex from the trace; see §4.3). When similarity is high, regress `x_s → x_t`. When similarity is low, regress `x_s` toward the trajectory implied by `c_s`'s action label, looked up from a teacher-derived prototype table.

This is the most under-specified piece of the project and the agent should expect to iterate. Build the canonical action vocabulary first (§4.3), then the loss.

Default δ = 0.3.

### 4.3 Canonical action vocabulary (prerequisite work)

Before Stage 4 training starts:

1. Parse all teacher CoC traces in the train set.
2. Extract action verbs/phrases via regex over the verb-object slots ("nudge to the right", "yield to pedestrian", "follow lead vehicle", etc.).
3. Cluster into ~20 canonical actions. Manual review required.
4. For each canonical action, compute the centroid teacher trajectory across all clips that use it. Store as `outputs/stage4/action_prototypes.npz`.

This is a half-day of work and unblocks the consistency loss. If the agent finds fewer than ~12 distinct actions, re-examine — the dataset may be too narrow for Stage 4 to add value, in which case Stage 4 reduces to Stage 2 + Stage 3 joint fine-tuning with δ = 0 and that should be flagged.

### 4.4 Training loop

- Optimizer: AdamW, lr 5e-5 (lower than Stages 2/3 because we're refining, not learning from scratch), cosine schedule, 3% warmup.
- Batch: 1 clip / 4 trajectories per step, gradient accumulation 8.
- Both gradient checkpointing AND CPU-offloaded optimizer states. This stage is the tightest on memory; expect to spend a day debugging OOMs.
- Train 5 epochs on the 80-clip train set.

### 4.5 Outputs

- `outputs/stage4/student_full/` — final LoRA adapter, hidden-state adapter, Action Expert, and `action_prototypes.npz`.
- `outputs/stage4/training_curves.json` — every loss component per step.

### 4.6 Acceptance criteria

| Criterion | Target |
|-----------|--------|
| ADE/FDE on val improved or within 5% of Stage 3 | passes |
| Reasoning–trajectory consistency rate (student's CoC action label matches the action label implied by predicted trajectory, via the prototype table) | ≥ 85% |
| No regression in CoC top-1 token agreement | within 3% of Stage 2 result |
| Training loss curves: all four components (CoC-KL, align, FM, consistency) trend down or stable | visual inspection, documented in PR |

If ADE/FDE regresses by more than 5% relative to Stage 3, revert to Stage 3 weights and ship those — Stage 4 is opt-in if it helps, not a mandatory step.

---

## 5. Evaluation Harness

**Test set:** the 10 held-out clips. **Comparison points:** teacher (Alpamayo-1.5), student-after-Stage-3, student-after-Stage-4.

### 5.1 Metrics

- **Trajectory error**: ADE, FDE at 1s / 3s / 6s horizons. Computed against teacher trajectory (this is a distillation eval, not a planner eval against a logged human driver).
- **Latency**: end-to-end (camera frames → emitted trajectory) on RTX 4090, batch 1, 100-trial median, warmup 10 trials. Report mean, median, p95.
- **Memory footprint**: peak GPU memory during inference, measured via `torch.cuda.max_memory_allocated()` over 100 calls.
- **Parameter count**: trainable, frozen, and total. Quantized weights count at their dequantized size for fair comparison.

### 5.2 Reproducibility

`scripts/eval.py --checkpoint <path> --hardware rtx4090` produces a single Markdown table that drops into the final report. Save raw per-clip numbers to `outputs/eval/<run_name>/raw.json`.

### 5.3 Acceptance criteria for the project

| Criterion | Target |
|-----------|--------|
| Total parameter count | ≤ 5.5 B |
| Peak inference VRAM, batch 1 | ≤ 24 GB |
| ADE @ 6s vs. teacher | ≤ 0.7 m |
| End-to-end latency, batch 1 | ≤ 200 ms |

These four together are the project's success definition.

---

## 6. Risks and Pre-Identified Pitfalls

These are issues to plan around, not problems to solve later.

The **hidden-state alignment** is the load-bearing assumption. If teacher hidden states are not slice-stable across runs (e.g., the teacher uses cached KV that varies with generation length), the cached states from Stage 1 will not match what the live teacher would produce now. Verify on day one by re-running teacher inference on three clips and diffing hidden states against the dump. If they differ, the dump must be regenerated with deterministic generation settings before Stage 2 begins.

**Single-step flow matching at this scale on driving trajectories** is not a settled recipe. Most flow-matching results in the literature are on images. Budget time for the regularization in §3.2 (`L_traj`) to be insufficient and for a 2-step or 4-step inference fallback to be needed. If 4-step is needed to meet ADE targets, the project still wins on memory and parameter count, but latency claims tighten.

**QLoRA on Qwen2.5-VL-3B** with vision tokens is memory-spiky on long video clips. If a clip has >150 frames, expect activation memory to balloon beyond gradient checkpointing's relief. Truncate or chunk clips at 128 frames and document the truncation in the eval report.

**Stage 4 is the riskiest stage and adds the least guaranteed value.** Build Stages 1–3 + the eval harness end-to-end first, ship a checkpoint, then attempt Stage 4. Do not let Stage 4 block reporting Stage 3 results.

**The "approximate RL" framing** in Stage 4 is a hand-wave over a real gap. If reviewers push back, the honest answer is that we are doing supervised consistency regularization, not RL, and that doing real RL (e.g., DPO over trajectory pairs) is future work.

---

## 7. Suggested Sequencing for Coding Agents

A reasonable assignment for parallel work:

- **Agent A** — Repo scaffolding (§0), data contract loader and validators (§1), eval harness skeleton (§5). Two days. Unblocks everyone.
- **Agent B** — Stage 2 (§2), starting once Agent A's loader passes its tests. Four to five days.
- **Agent C** — Stage 3 (§3), starting once Stage 2 has a checkpoint and hidden-state cache. Three to four days.
- **Agent D** — Stage 4 (§4), starting once Stage 3 acceptance criteria are met. Five to seven days, with the canonical-action-vocabulary work (§4.3) as the first milestone.

Stages 2/3 can overlap by one day if Agent C uses a synthetic hidden-state cache (random tensors of correct shape) to develop the Action Expert and loss in parallel.

---

## 8. Out of Scope

To avoid scope creep, the following are explicitly **not** part of this plan: deployment to the actual UB Autonomous Lincoln vehicle, ROS integration, sensor fusion beyond the cameras the teacher already uses, online learning or continual updates, and any safety certification. Those are downstream of this distillation work.
