#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_MODE=none "${SCRIPT_DIR}/shell.sh" python -m scripts.validate_teacher_dump \
    --root "${TEACHER_DUMP_ROOT:-data/teacher_dump}" \
    --splits "${SPLITS_PATH:-data/splits}" \
    "$@"
