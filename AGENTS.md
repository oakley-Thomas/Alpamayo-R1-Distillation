# AGENTS.md

Operational guidance for LLM coding agents working on this repo. **Read this file completely before starting any task.** It is short on purpose; everything in it is load-bearing.

For *what* to build, see [`README.md`](./README.md) and [`DISTILLATION_PLAN.md`](./DISTILLATION_PLAN.md). This file covers *how to work*.

---

## 1. Orientation (do this first, every time)

Before writing or modifying any code on a task:

1. Read the task description in full. Do not start implementing on the first sentence.
2. Read `DISTILLATION_PLAN.md` — at minimum, the section for the stage you're touching, plus §0 (Conventions) and §1 (Data Contract).
3. Run `git log --oneline -20` and `git status`. Know what's already been done and what's in flight.
4. Check `tests/` for existing tests covering the area you're about to change. Read them — they encode contracts the next agent (or you) will rely on.
5. Only then start planning the change.

If any of steps 1–4 reveal a contradiction with the task as written, stop and raise it (see §6). Do not paper over contradictions.

---

## 2. Code style and quality

**Language:** Python 3.11+. No JS, no Bash beyond thin entry-point scripts, no notebooks committed to the repo (notebooks are fine for exploration, but extract the code into `src/` before opening a PR).

**Formatters and linters (all must pass before commit):**

- `ruff format .` — formatter, replaces black.
- `ruff check . --fix` — linter, full default rule set plus `I` (import sorting), `B` (bugbear), `UP` (pyupgrade), `SIM` (simplify), `RUF` (ruff-specific).
- `pyright src/ tests/` — strict mode. No `# type: ignore` without a comment explaining why.

These are run by `make lint` and enforced by CI. Do not commit code that fails them. If a rule fights you on something genuinely necessary, disable it for that line with a justification comment, not project-wide.

**Imports:** absolute imports rooted at `src/` only. No `from ..foo import bar`. No star imports.

**Type hints:** mandatory on every function signature, including private ones. Use `from __future__ import annotations` at the top of every file. Tensor shapes go in the docstring, not as comments — see the docstring convention below.

**Docstrings:** every public function, class, and module gets one. PyTorch nn.Module subclasses must document tensor shapes for `forward`:

```python
def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Predict velocity at noise level t.

    Args:
        x: noisy trajectory, shape (B, 64, 3) — batch, waypoints, (x, y, heading).
        t: noise level in [0, 1], shape (B,).

    Returns:
        velocity field prediction, shape (B, 64, 3).
    """
```

If shapes are wrong at runtime, this docstring is the contract that gets blamed. Get it right.

**No magic numbers in training code.** Hyperparameters live in `configs/*.yaml`, loaded into a typed dataclass. The training script reads from the dataclass, never from a literal. The one exception is mathematical constants (π, ε for numerical stability) — those go in code with a comment.

**Error handling:** fail loudly. No `try/except: pass`. No silent fallbacks (e.g., "if CUDA isn't available, use CPU"). If you catch an exception, you re-raise with context, log and re-raise, or handle it in a way that leaves the system in a known state. Data-loading errors must include the offending clip ID. Loss values that go to NaN must crash the run, not get masked to zero.

---

## 3. Testing

**Tests are required for every PR, not just per stage.** The plan's stage-level acceptance criteria are necessary, not sufficient.

**What gets a test:**

- Every new function with non-trivial logic (more than one branch, or any tensor reshaping).
- Every loss component — at minimum, a test that it's finite on a synthetic batch and zero (or near-zero) when student output equals teacher output.
- Every data loader — a test on a tiny fixture in `tests/fixtures/` that exercises the happy path and at least one corruption mode.
- Every model `forward` — a shape test with batch size 2.

**What doesn't need a test:**

- Pure config plumbing (loading YAML into a dataclass).
- One-off scripts in `scripts/` that aren't on the training path.
- Visualization / plotting code.

**Test conventions:**

- `pytest` only. No `unittest`.
- Tests live in `tests/`, mirroring the `src/` layout: `src/losses/coc_kl.py` → `tests/losses/test_coc_kl.py`.
- Tests use CPU and tiny tensors by default. GPU-only tests are marked `@pytest.mark.gpu` and skipped in CI's CPU lane.
- A test that requires real teacher-dump data is marked `@pytest.mark.requires_dump` and uses `tests/fixtures/mini_dump/` (a 2-clip fixture committed to the repo, not real data).
- Run all tests with `pytest -x` before opening a PR. `-x` is required so you fix the first failure, not skim past it.

**Acceptance criteria for stage merges** (in addition to the plan's): all stage-relevant tests pass, the eval harness produces numbers that hit the §2.5 / §3.5 / §4.6 targets in the plan, and the run is reproducible from `run_config.yaml` + a fresh checkout.

---

## 4. Git workflow

**Branches:** one branch per task. Naming: `<stage>/<short-desc>`, e.g., `stage2/hidden-state-adapter`, `eval/latency-harness`, `infra/data-loader`.

**Commits: Conventional Commits, strictly.**

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`, `build`, `ci`. Scopes match top-level `src/` directories: `data`, `models`, `losses`, `train`, `eval`, plus `infra` for repo-level changes.

Subject line: imperative mood, no period, ≤ 72 characters. Body: wrap at 80, explain *why* not *what*. Footer: `Closes #N` for issues, `Co-authored-by:` for collaboration.

Examples:

```
feat(losses): add hidden-state alignment loss for Stage 2

Implements smooth-L1 alignment between student and teacher hidden
states at Action-Expert-conditioning positions. Smooth-L1 chosen over
MSE because magnitude outliers at scene transitions over-weight MSE.

Closes #14
```

```
fix(data): raise on missing hidden_states.npy instead of zero-filling

Silent zero-fill was masking corrupt teacher dumps and producing
plausible-but-wrong Stage 2 results. Now raises with clip ID.
```

**Commit hygiene:**

- One logical change per commit. If your commit message contains "and," you probably want two commits.
- Squash WIP commits before opening a PR.
- Never commit directly to `main`. Never force-push to a shared branch.
- Never commit checkpoints, dataset files, `.env`, or anything in `outputs/` or `data/teacher_dump/`. The `.gitignore` covers these — if you find yourself running `git add -f`, stop.

**Pull requests:**

- Title: same format as the commit subject line.
- Description must include: what changed, why, how it was tested, and which acceptance criteria from the plan it satisfies (if any).
- A PR that touches a stage's training loop must include a screenshot or paste of the loss curves from a short training run, even a 50-step smoke test.
- PRs touching `losses/` or `models/` require a passing test that exercises the new code path. No exceptions.

---

## 5. Working with the data and models

**Teacher dump is read-only.** Stages 2–4 read from `data/teacher_dump/` and never write to it. If a teacher dump entry looks wrong, the fix is in Stage 1 (re-run inference with corrected settings), not in a downstream loader's "cleanup" step.

**Memory is the dominant constraint.** Before adding any tensor allocation in a training loop, ask: does this need to be on GPU? Does this need full precision? Does this need to live across iterations? The single 48 GB card is unforgiving — code that works on a beefy multi-GPU node will OOM here.

**No silent device or dtype changes.** If you call `.to(device)` or `.half()` or `.float()` somewhere unexpected, document it. Mixed-precision boundaries are bug nests.

**Determinism.** Set seeds (`torch`, `numpy`, `random`, `torch.cuda`) at the top of every training script via the shared `src/utils/seed.py` helper. Use `torch.use_deterministic_algorithms(True)` where feasible. If a kernel forces non-determinism, document it in the run's notes.

**Checkpointing.** Save checkpoints with the optimizer state, scheduler state, RNG state, and a copy of the config used to produce them. A checkpoint that can't be resumed is not a checkpoint.

---

## 6. Judgment: when to push through, when to stop

Strict rules cover most situations. The rest is judgment. Use the following calibration.

**Push through (decide, document, keep moving) when:**

- The task is well-specified, you have a reasonable approach, and the only uncertainty is a low-stakes implementation detail (e.g., variable name, helper-function placement, choice between two equivalent library APIs).
- A linter or test fails and the fix is obvious from the error message.
- The plan specifies a default value (e.g., α = 1.0) and you're considering minor adjustments within an order of magnitude — note the change in the PR.
- You discover a small bug adjacent to your task. Fix it in a separate commit on the same branch, with a `fix(...)` message.

**Stop and raise when:**

- The task contradicts the plan, the data contract, or another stage's outputs.
- You're about to weaken an acceptance criterion to make a test pass.
- You're about to add a `# type: ignore`, a bare `except`, or a silent fallback because the principled fix is hard.
- You discover the teacher dump itself is malformed (this is a Stage 1 problem, not yours to patch around).
- You need to change a public interface another stage depends on (loss signatures, model `forward` shapes, dataset return types).
- The hardware genuinely can't run what the plan asks for, even after applying the plan's memory-reduction guidance.
- A core assumption in the plan (e.g., "teacher hidden states are deterministic across runs") turns out to be false.

The rule of thumb: if your fix is making the symptom go away rather than addressing the cause, stop.

**How to raise:** open a GitHub issue with the `blocked` label, link it from your PR, summarize the contradiction in 3–5 sentences, and propose 2 options. Then *wait* — don't pick one and proceed. While waiting, work on a different task or branch.

**What "stop" doesn't mean:** stop ≠ delete your work. Push the WIP branch, mark the PR as draft, leave a comment with where you left off. The next agent (or you, after the human responds) should be able to pick it up.

---

## 7. Communication norms

**Comments in code:** explain *why*, not *what*. Code says what; comments say why a non-obvious choice was made or what gotcha is being avoided. Avoid:

```python
# Increment counter
counter += 1
```

Prefer:

```python
# We count valid frames, not all frames, because masked frames
# would otherwise dominate the alignment loss on long clips.
counter += 1
```

**TODO comments:** allowed only with an issue reference: `# TODO(#42): switch to fused kernel once ROCm 6.2 lands`. Bare `# TODO` will fail lint.

**PR review responses:** if a reviewer asks why you did something, the answer goes in the code or commit message, not just in the PR thread. If it's only in the thread, the next person to read the code won't see it.

**Don't claim work is done if it isn't.** If acceptance criteria aren't met, say so in the PR description. Half-complete work clearly labeled is more useful than complete-looking work that fails on the first real test.

---

## 8. Things this project specifically cares about

These come up often enough on Alpamayo distillation work to call out:

- **Hidden-state slicing must match the teacher exactly.** This is the load-bearing assumption of Stage 2. If you change how positions are selected, you've silently changed what the Action Expert sees. See `src/models/teacher_iface.py` and the plan §2.1.
- **Trajectory normalization stats travel with the checkpoint.** Don't compute them on the fly at inference. The Action Expert was trained with specific per-axis mean/std; using different stats at inference produces wrong but plausible-looking trajectories.
- **Don't conflate "diffusion" and "flow matching" in code or comments.** They're related but not the same; the student is flow matching specifically, and the loss / inference path differ.
- **The Stage 4 consistency loss is a known-fragile component.** If you're touching it and things look off, that's expected — raise via §6 rather than tweaking δ until the loss curve looks pretty.

---

## 9. Out of scope for any agent

The following are not yours to change without explicit human direction:

- The teacher model or Stage 1 inference code.
- The acceptance criteria in `DISTILLATION_PLAN.md`. (You can argue they're wrong via §6, but you don't unilaterally lower them.)
- The repo layout in `DISTILLATION_PLAN.md` §0. New top-level directories require an issue and approval.
- The hardware target (single 24–48 GB GPU). Code that assumes multi-GPU is out of scope.
- The choice of student architecture (Qwen2.5-VL-3B + flow-matching Action Expert). Substituting a different backbone is a project-level decision, not an agent decision.

---

## 10. Quick checklist before opening a PR

- [ ] `ruff format .` clean
- [ ] `ruff check .` clean
- [ ] `pyright src/ tests/` clean
- [ ] `pytest -x` passes
- [ ] New code has tests (see §3)
- [ ] Docstrings on new public functions, with tensor shapes for `forward` methods
- [ ] No magic numbers in training code (config-driven)
- [ ] Commit messages follow Conventional Commits
- [ ] No checkpoints, data files, or secrets committed
- [ ] PR description names the acceptance criteria addressed (if any)
- [ ] If a stage training loop changed, loss-curve evidence included
- [ ] If blocked on anything, issue opened and linked
