#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-alpamayo-distill:stage2}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}"
HF_VOLUME="${HF_VOLUME:-}"
GPU_MODE="${GPU_MODE:-all}"

if [[ -n "${HF_VOLUME}" ]]; then
    HF_MOUNT_SOURCE="${HF_VOLUME}"
else
    HF_MOUNT_SOURCE="${HF_CACHE}"
    mkdir -p "${HF_CACHE}"
fi

mkdir -p "${REPO_ROOT}/data" "${REPO_ROOT}/outputs"

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

ENV_ARGS=(
    -e HF_HOME=/cache/huggingface
    -e "HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}"
    -e PYTHONPATH=/workspace:/workspace/alpamayo1.5/src
)
if [[ -n "${HF_TOKEN:-}" ]]; then
    ENV_ARGS+=(-e "HF_TOKEN=${HF_TOKEN}")
fi
if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    ENV_ARGS+=(-e "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}")
fi

docker run --rm "${TTY_ARGS[@]}" \
    "${GPU_ARGS[@]}" \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    "${ENV_ARGS[@]}" \
    -v "${REPO_ROOT}:/workspace" \
    -v "${HF_MOUNT_SOURCE}:/cache/huggingface" \
    -w /workspace \
    "${IMAGE_NAME}" \
    "$@"
