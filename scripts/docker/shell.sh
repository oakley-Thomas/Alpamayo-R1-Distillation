#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-alpamayo-distill:stage2}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}"
GPU_MODE="${GPU_MODE:-all}"

mkdir -p "${HF_CACHE}" "${REPO_ROOT}/data" "${REPO_ROOT}/outputs"

if [ "$#" -eq 0 ]; then
    set -- /bin/bash
fi

TTY_ARGS=()
if [ -t 0 ] && [ -t 1 ]; then
    TTY_ARGS=(-it)
fi

GPU_ARGS=()
case "${GPU_MODE}" in
    all)
        GPU_ARGS=(--gpus all)
        ;;
    none)
        ;;
    *)
        echo "Unsupported GPU_MODE: ${GPU_MODE}" >&2
        exit 2
        ;;
esac

docker run --rm "${TTY_ARGS[@]}" \
    "${GPU_ARGS[@]}" \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e HF_HOME=/cache/huggingface \
    -e PYTHONPATH=/workspace:/workspace/alpamayo1.5/src \
    -v "${REPO_ROOT}:/workspace" \
    -v "${HF_CACHE}:/cache/huggingface" \
    -w /workspace \
    "${IMAGE_NAME}" \
    "$@"
