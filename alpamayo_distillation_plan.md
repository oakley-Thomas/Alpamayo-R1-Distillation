# Alpamayo-1.5 → Student VLA Distillation: Implementation Plan

**Target:** Reasoning + trajectory VLA student, 2–4B params, onboard real-time
**Teacher:** `nvidia/Alpamayo-1.5-10B` (Cosmos-Reason2 backbone + diffusion trajectory decoder)
**Supervision:** Logit KD (KL on CoT + trajectory tokens) + output mimicry (trajectory L2)
**Eval:** Open-loop minADE / minFDE / RMSE; latency on RTX 5080

> **Legend:**
> 🟢 = LLM implements directly
> 🟡 = LLM implements, but with documented assumption that human should verify
> 🔴 = **HUMAN DECISION/ACTION REQUIRED** before LLM proceeds

---

## Phase 0: Pre-implementation decisions (do these first)

### ✅ H0.1 — Confirm tokenizer strategy — **RESOLVED**
Cosmos-Reason2 (Alpamayo-1.5's backbone) is post-trained from `Qwen3-VL`, so its tokenizer is the Qwen3 tokenizer — publicly available and shared with the student backbone.
- **Decision:** `shared_tokenizer` — logit KD is valid with direct index alignment. No remapping or ULD needed.
- **Committed:** `configs/distill.yaml` → `kd.vocab_strategy: "shared_tokenizer"`
- **Processor to use:** `AutoProcessor.from_pretrained("nvidia/Cosmos-Reason2-2B")`

### ✅ H0.2 — Choose student LLM backbone — **RESOLVED**
Since Cosmos-Reason2-2B is a fine-tune of `Qwen3-VL-2B-Instruct`, using the same base model as the student gives identical tokenizer, vocabulary, architecture, and positional encoding — the ideal setup for logit KD.
- **Decision:** `Qwen/Qwen3-VL-2B-Instruct`
- **Committed:** `configs/student.yaml` → `backbone.name: "Qwen/Qwen3-VL-2B-Instruct"`
- **Note:** Qwen3-VL has a built-in vision encoder + MLP projector. The separate SigLIP encoder and Q-Former resampler are not needed; `vision_encoder.source` is set to `"backbone"` and `resampler.enabled` is `false`.

### 🔴 H0.3 — Action decoder choice
- **Option A:** Flow matching, 1–4 sampling steps (recommended balance).
- **Option B:** Deterministic MLP over learned trajectory codebook (fastest).
- **Option C:** Distill teacher's diffusion to a consistency model.
- **Action:** Commit to `configs/student.yaml` under `action_decoder.type`. Default: flow matching.

### 🔴 H0.4 — Dataset selection
- **Action:** Confirm which datasets are available locally on `cavas-skytech` or RTX 5080 machine. Candidates: NVIDIA Physical AI dataset subset, NuScenes, internal CAVAS captures.
- **Action:** Document paths and sizes in `configs/data.yaml`.

### 🔴 H0.5 — Compute budget confirmation
- Teacher inference (10B model, bf16) needs ~22GB VRAM — fits on RTX 5080 (16GB? confirm) or Quadro RTX 8000 (48GB).
- **Action:** Confirm which machine runs teacher inference (data generation) vs. student training. Recommendation: teacher rollouts on Quadro RTX 8000, student training on RTX 5080.

---

## Phase 1: Repository scaffolding

### 🟢 T1.1 — Project structure
```
alpamayo_distill/
├── configs/
│   ├── data.yaml
│   ├── distill.yaml
│   ├── student.yaml
│   └── train_stage{1,2,3,4}.yaml
├── src/
│   ├── data/
│   │   ├── teacher_rollout.py     # Generates teacher outputs offline
│   │   ├── dataset.py              # Loads cached teacher outputs + GT
│   │   └── tokenizer_utils.py
│   ├── models/
│   │   ├── student_vla.py          # Full student model
│   │   ├── vision_encoder.py
│   │   ├── action_decoder.py
│   │   └── resampler.py            # Q-Former or token-merge for visual token reduction
│   ├── losses/
│   │   ├── kd_loss.py              # KL with temperature, top-k support
│   │   ├── trajectory_loss.py
│   │   └── selective_kd.py         # Token-importance weighting
│   ├── training/
│   │   ├── stage1_warmup.py
│   │   ├── stage2_action.py
│   │   ├── stage3_joint.py
│   │   └── stage4_rl.py            # Optional
│   ├── eval/
│   │   ├── open_loop.py            # minADE/minFDE/RMSE
│   │   ├── latency.py
│   │   └── alignment.py            # CoT-trajectory alignment scoring
│   └── utils/
│       ├── logits_storage.py       # int16 top-k caching
│       └── viz.py
├── scripts/
│   ├── run_teacher_rollout.sh
│   ├── train_stage{1,2,3,4}.sh
│   └── eval_all.sh
├── notebooks/
│   └── inspect_teacher_outputs.ipynb
└── pyproject.toml
```

### 🟢 T1.2 — Environment setup
- Python 3.11, PyTorch 2.4+, CUDA 12.x, bf16 support.
- Dependencies: `transformers`, `accelerate`, `deepspeed` (optional), `flash-attn` (verify Blackwell wheel availability), `hydra-core`, `wandb`, `einops`, `numpy`, `scipy`.
- 🟡 **Note for LLM:** Add `flash-attn` install with fallback to SDPA if Blackwell wheel unavailable. Document this clearly.

---

## Phase 2: Teacher rollout pipeline (Week 1, days 1–3)

### 🟢 T2.1 — Implement `teacher_rollout.py`
For each scene in the dataset:
1. Load multi-camera frames + egomotion history (4 cams, 4 frames at 10Hz, 1080×1920 → 320×576).
2. Run Alpamayo-1.5 inference with deterministic seed.
3. Capture and serialize:
   - Sampled token sequence (CoT + trajectory tokens), full
   - **Top-k logits per generated token** (k=32 default, configurable)
   - Final decoded trajectory (N×T waypoints with heading)
   - Egomotion history (input)
   - Scene metadata (id, timestamp, source dataset)
4. Optional: capture hidden states from 2–3 specified transformer layers for future feature-KD experiments.

### 🟢 T2.2 — Implement `logits_storage.py`
- Store top-k as `int16` + softmax temperature scalar, with sparse indices.
- Compression target: ~10× vs fp32.
- Format: HDF5 or memory-mapped numpy.
- Include CRC32 checksum per scene.

### 🟡 T2.3 — Validation script
After rollout, sample 20 random scenes and verify:
- Decoded trajectories from cached tokens match teacher's online output (sanity check).
- Top-k logits sum to a reasonable mass (>0.95 typically).
- Storage size matches expectations (flag if blow-up).

### 🔴 H2.4 — Run rollout at scale
- **Action:** Schedule rollout job. Estimate: 99ms/scene → ~10 scenes/sec → 100K scenes ≈ 3 hours of GPU time, but realistically 2–3× that with I/O.
- **Action:** Decide rollout dataset size. Recommendation: start with 10K scenes for pipeline validation, then scale to 100K–500K for final training.

---

## Phase 3: Student model implementation (Week 1, days 4–7)

### 🟢 T3.1 — Vision encoder
- Qwen3-VL-2B-Instruct has a built-in ViT-based vision encoder + MLP projector. Use it directly — no separate SigLIP or Q-Former needed.
- Configure to accept 4-camera inputs: process each camera independently through the vision encoder, then concatenate visual tokens before the LLM.
- Set `vision_encoder.source: "backbone"` and `resampler.enabled: false` in `configs/student.yaml` (already set).

### 🟢 T3.2 — Egomotion encoder
- Small MLP: input = 3D translation + 9D rotation × N timesteps → 4 prefix tokens.

### 🟢 T3.3 — Backbone integration
- Load `Qwen/Qwen3-VL-2B-Instruct` via `Qwen3VLForConditionalGeneration.from_pretrained(...)`.
- Use `AutoProcessor.from_pretrained("nvidia/Cosmos-Reason2-2B")` for the processor — this guarantees vocab alignment with the teacher.
- Inject egomotion tokens as prefix tokens ahead of visual tokens in the input sequence.
- Add LoRA adapters (r=32, alpha=64) for stages 1 & 3 to reduce memory footprint.

### 🟢 T3.4 — Action decoder
Implement based on H0.3:
- **Flow matching**: small DiT-style transformer, condition on final hidden state of LLM, 1–4 ODE steps at inference.
- **MLP+codebook**: VQ-VAE-style trajectory codebook (learned offline from GT trajectories), then classification + residual regression.
- Output: N×T waypoints (x, y, heading) for planning horizon T (default 6 seconds at 2Hz = 12 waypoints).

### 🟡 T3.5 — Forward pass contract
- Input dict: `{cameras: [B,4,3,H,W], egomotion: [B,N,12], text: [B,L]}`
- Output dict: `{logits: [B,L,V], trajectory: [B,T,3], cot_text: list[str]}`
- 🟡 **LLM should write a unit test** that runs forward + backward on a tiny batch with random data before integrating with real data.

---

## Phase 4: Loss implementation (Week 2, days 1–2)

### 🟢 T4.1 — Logit KD loss
```python
# Pseudocode
def kd_loss(student_logits, teacher_topk_logits, teacher_topk_indices, temperature=2.0):
    # Gather student logits at teacher's top-k indices
    # Apply temperature softmax to both
    # Compute KL(teacher || student) per position, average over sequence
    # Multiply by temperature^2
    return loss
```
- Support for both full-vocab teacher logits (if stored) and top-k mode.
- Ignore padding tokens via mask.

### 🟢 T4.2 — Trajectory L2 loss
- Position L2 + heading L2 with `lambda_head=0.1` default.
- Average over horizon, weight later timesteps slightly less (e.g., exponential decay 0.95).

### 🟢 T4.3 — Ground-truth anchor (minADE)
- Standard minADE over sampled trajectories (k=1 and k=6 modes).

### 🟡 T4.4 — Selective KD weighting
- Default to uniform weighting initially.
- Implement infrastructure for token-importance weights (computed offline from teacher attention or trajectory mutual info).
- 🟡 **Mark this as v2** — get baseline working first, ablate later.

### 🟢 T4.5 — Combined loss module
```python
total = alpha * L_KD + beta * L_traj + gamma * L_GT
```
- Defaults: α=1.0, β=1.0, γ=0.5. Expose all in config.

---

## Phase 5: Training stages

### 🟢 T5.1 — Stage 1: VLM warm-up (Week 2, days 3–5)
- Freeze action decoder.
- Train vision encoder + resampler + LLM (with LoRA) on CoT-only KD.
- LR: 5e-5 (LoRA) / 1e-5 (full).
- ~1 epoch over rollout data.
- Checkpoint: `ckpts/stage1_warmup.pt`.

### 🟢 T5.2 — Stage 2: Action head training (Week 2, days 6–7 + Week 3, days 1–2)
- Freeze backbone.
- Train action decoder on `L_traj + L_GT`.
- LR: 1e-4.
- ~2 epochs.
- Checkpoint: `ckpts/stage2_action.pt`.

### 🟢 T5.3 — Stage 3: Joint fine-tune (Week 3, days 3–7)
- Unfreeze everything.
- Full combined loss.
- LR: 1e-5 with cosine decay.
- Mixed precision bf16.
- Gradient checkpointing on backbone.
- ~3 epochs.
- Checkpoint: `ckpts/stage3_joint.pt`.

### 🔴 H5.4 — Stage 4: RL post-training (Week 4, optional)
- Only if compute and timeline allow.
- **Action:** Decide whether to skip. Default: skip for v1, plan for v2.

---

## Phase 6: Evaluation (Week 4)

### 🟢 T6.1 — Open-loop metrics
- minADE @ k=1, k=6
- minFDE @ k=1, k=6
- RMSE: total, longitudinal, lateral (lateral matters more for safety)
- Report on held-out split.

### 🟢 T6.2 — Latency benchmark
- Measure on RTX 5080 in fp16 and int8.
- Warmup 10 iters, time 100 iters, report p50/p95.
- **Include:** tokenization + vision encode + LLM forward + action decode. Exclude data loading.

### 🟢 T6.3 — Reasoning-trajectory alignment
- For each sample, generate CoT + trajectory.
- Use teacher (or a small judge model) to classify whether trajectory matches stated intent in CoT.
- Report alignment % over eval set.

### 🟡 T6.4 — Compare against teacher
- Run same metrics on Alpamayo-1.5.
- Compute student/teacher ratio per metric.
- Flag any metric where student < 80% of teacher.

### 🔴 H6.5 — Closed-loop AlpaSim eval
- **Action:** Decide if AlpaSim integration is in scope for v1. The open-loop → closed-loop gap often reveals distillation failures.
- If yes: integrate student into AlpaSim's evaluation harness, run on standard scene set, report AlpaSim score.

---

## Phase 7: Deliverables checklist

### 🟢 T7.1 — Code
- All training scripts runnable end-to-end with `bash scripts/train_stageN.sh`.
- Config-driven (Hydra), reproducible (seed in every config).

### 🟢 T7.2 — Final report sections
- Architecture diagram
- Loss formulation (with equations)
- Training curves per stage (W&B export)
- Eval table: student vs. teacher across all metrics
- Latency table
- Ablations: KD weight α, temperature τ, with/without CoT KD

### 🟢 T7.3 — Model artifact
- Final student weights (bf16 and int8 quantized).
- Inference script with example input.
- Model card following NVIDIA Alpamayo's template.

---

## Risks the LLM should watch for during implementation

1. **Tokenizer mismatch silently corrupting KD** — verify on day one with a tiny end-to-end test (T3.5 unit test).
2. **Diffusion KL ≠ categorical KL** — never apply vanilla KL to the action decoder's diffusion process. Trajectory L2 only.
3. **Visual token explosion** — 4 cameras × full Qwen3-VL visual tokens may be heavy. Monitor token count per forward pass; if OOM, reduce input resolution or enable dynamic resolution cropping.
4. **Cached logits format change mid-training** — version the storage format and refuse to load mismatched versions.
5. **Flash-attention Blackwell compatibility** — fall back to SDPA if wheels unavailable. Don't silently lose attention.
6. **CoT length variance** — use bucketed sampling or pad-to-max-with-masking. Don't truncate, you'll lose decision tokens.

---

## Suggested LLM workflow

1. Read this plan top to bottom.
2. **Stop at every 🔴 and request the human decision before proceeding.** Do not guess.
3. For 🟡 items, implement with the documented default but flag the assumption in code comments and in the PR description.
4. Implement phases sequentially. Do not start Phase N+1 until Phase N has a passing smoke test.
5. After each phase, write a brief status summary: what was built, what was assumed, what needs human review.
