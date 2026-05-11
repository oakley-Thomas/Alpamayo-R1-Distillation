#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/shell.sh" python -m scripts.train_stage2 \
    --config "${STAGE2_CONFIG:-configs/stage2.yaml}" \
    "$@"
