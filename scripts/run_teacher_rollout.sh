#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"
python -m alpamayo_distill.data.teacher_rollout --data-config configs/data.yaml --distill-config configs/distill.yaml "$@"
