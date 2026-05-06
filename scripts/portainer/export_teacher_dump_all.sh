#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN must be set with access to Alpamayo-1.5-10B and PhysicalAI AV." >&2
    exit 2
fi

TEACHER_DUMP_ROOT="${TEACHER_DUMP_ROOT:-data/teacher_dump}"
SPLITS_PATH="${SPLITS_PATH:-data/splits}"
MODEL_NAME="${MODEL_NAME:-nvidia/Alpamayo-1.5-10B}"
NUM_TRAJ_SAMPLES="${NUM_TRAJ_SAMPLES:-16}"
TOP_K="${TOP_K:-32}"
MAX_GENERATION_LENGTH="${MAX_GENERATION_LENGTH:-256}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.98}"
T0_US="${T0_US:-5100000}"
INCLUDE_KV_CACHE="${INCLUDE_KV_CACHE:-true}"
CAPTURE_DENOISING="${CAPTURE_DENOISING:-true}"
STAGE2_ONLY="${STAGE2_ONLY:-false}"
OVERWRITE="${OVERWRITE:-false}"
VALIDATE_AFTER_EXPORT="${VALIDATE_AFTER_EXPORT:-true}"

split_files=()
if [[ -n "${SPLITS:-}" ]]; then
    IFS=',' read -r -a split_files <<< "${SPLITS}"
elif [[ -f "${SPLITS_PATH}" ]]; then
    split_files=("${SPLITS_PATH}")
else
    for split_name in train val test; do
        split_file="${SPLITS_PATH}/${split_name}.json"
        if [[ -f "${split_file}" ]]; then
            split_files+=("${split_file}")
        fi
    done
fi

if [[ "${#split_files[@]}" -eq 0 ]]; then
    echo "No split files found. Mount them under ${SPLITS_PATH} or set SPLITS." >&2
    exit 2
fi

common_args=(
    --output-root "${TEACHER_DUMP_ROOT}"
    --model-name "${MODEL_NAME}"
    --num-traj-samples "${NUM_TRAJ_SAMPLES}"
    --top-k "${TOP_K}"
    --max-generation-length "${MAX_GENERATION_LENGTH}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --t0-us "${T0_US}"
)

if [[ "${INCLUDE_KV_CACHE}" == "true" ]]; then
    common_args+=(--include-kv-cache)
fi
if [[ "${CAPTURE_DENOISING}" != "true" ]]; then
    common_args+=(--no-capture-denoising)
fi
if [[ "${STAGE2_ONLY}" == "true" ]]; then
    common_args+=(--stage2-only)
fi
if [[ "${OVERWRITE}" == "true" ]]; then
    common_args+=(--overwrite)
fi

for split_file in "${split_files[@]}"; do
    echo "Exporting teacher dump for ${split_file}"
    python -m scripts.export_teacher_dump \
        --clip-ids "${split_file}" \
        "${common_args[@]}"
done

if [[ "${VALIDATE_AFTER_EXPORT}" == "true" ]]; then
    validate_args=(--root "${TEACHER_DUMP_ROOT}")
    if [[ "${INCLUDE_KV_CACHE}" == "true" ]]; then
        validate_args+=(--include-kv-cache)
    fi
    for split_file in "${split_files[@]}"; do
        echo "Validating teacher dump for ${split_file}"
        python -m scripts.validate_teacher_dump \
            "${validate_args[@]}" \
            --splits "${split_file}"
    done
fi
