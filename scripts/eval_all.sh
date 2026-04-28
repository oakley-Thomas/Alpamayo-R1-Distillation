#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"
python -m unittest discover -s tests "$@"
