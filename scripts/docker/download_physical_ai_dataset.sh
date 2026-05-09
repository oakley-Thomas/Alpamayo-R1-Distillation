#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-alpamayo-distill:stage2}"
HF_VOLUME="${HF_VOLUME:-NVIDIA-PHYSICAL-AI}"
DATASET_REPO_ID="${DATASET_REPO_ID:-nvidia/PhysicalAI-Autonomous-Vehicles}"

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    echo "HF_TOKEN is not set; gated dataset download will fail unless the mounted volume already has a Hugging Face login." >&2
fi

ENV_ARGS=(-e HF_HOME=/cache/huggingface)
if [[ -n "${HF_TOKEN:-}" ]]; then
    ENV_ARGS+=(-e "HF_TOKEN=${HF_TOKEN}")
fi
if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    ENV_ARGS+=(-e "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}")
fi

docker run --rm \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    "${ENV_ARGS[@]}" \
    -v "${REPO_ROOT}:/workspace" \
    -v "${HF_VOLUME}:/cache/huggingface" \
    -w /workspace \
    "${IMAGE_NAME}" \
    python -m scripts.download_physical_ai_dataset \
        --repo-id "${DATASET_REPO_ID}" \
        "$@"
