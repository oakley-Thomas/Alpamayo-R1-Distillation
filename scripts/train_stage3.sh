#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"
python -m alpamayo_distill.training.stage3_joint "$@"
