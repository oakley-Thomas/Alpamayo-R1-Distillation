#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/shell.sh" python -m scripts.export_teacher_dump \
    --clip-ids "${CLIP_IDS:-data/splits/train.json}" \
    --output-root "${TEACHER_DUMP_ROOT:-data/teacher_dump}" \
    --num-traj-samples "${NUM_TRAJ_SAMPLES:-16}" \
    --traj-sample-batch-size "${TRAJ_SAMPLE_BATCH_SIZE:-1}" \
    --top-k "${TOP_K:-32}" \
    "$@"
