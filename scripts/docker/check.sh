#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_MODE=none "${SCRIPT_DIR}/shell.sh" bash -lc \
    'ruff format --check . && ruff check . && pyright src tests && pytest -x'
