#!/usr/bin/env bash
set -euo pipefail

# Visual Enhancement few-shot evaluation for Qwen3-VL-4B.
# VE draws red GT boxes on support images only; query images remain untouched.
# The evaluator's fixed protocol is already inst/class-wise/positive-query.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-${CKPT:-${ROOT_DIR}/weights/Qwen3-VL-4B-Instruct}}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data}"
WORK_ROOT="${WORK_ROOT:-${OUTPUT_DIR:-${ROOT_DIR}/work_dirs/qwen3-vl-4b-ve-4gpu}}"
DEFAULT_DATASETS=(ArTaxOr clipart1k FISH NEU-DET UODD VISUALDIOR)
DEFAULT_SHOTS=(1 2 4)
DEVICE="${DEVICE:-cuda:0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MIN_PIXELS="${MIN_PIXELS:-3136}"
MAX_IMAGE_PIXELS="${MAX_IMAGE_PIXELS:-${MAX_PIXELS:-$((800 * 800))}}"
MIN_BOX_AREA_RATIO="${MIN_BOX_AREA_RATIO:-0}"
ATTENTION="${ATTENTION:-sdpa}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"

if [[ -n "${DATASETS:-}" ]]; then
  read -r -a DATASET_LIST <<< "${DATASETS}"
else
  DATASET_LIST=("${DEFAULT_DATASETS[@]}")
fi
if [[ -n "${SHOTS:-}" ]]; then
  read -r -a SHOT_LIST <<< "${SHOTS}"
else
  SHOT_LIST=("${DEFAULT_SHOTS[@]}")
fi

cd "${ROOT_DIR}"
mkdir -p "${WORK_ROOT}"
LOG_FILE="${LOG_FILE:-${WORK_ROOT}/evaluation.log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "model_path=${MODEL_PATH}"
echo "data_root=${DATA_ROOT}"
echo "work_root=${WORK_ROOT}"
printf 'datasets='
printf '%s ' "${DATASET_LIST[@]}"
printf '\n'
printf 'shots='
printf '%s ' "${SHOT_LIST[@]}"
printf '\n'
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "num_gpus=${NUM_GPUS}"
echo "batch_size=${BATCH_SIZE}"
echo "max_image_pixels=${MAX_IMAGE_PIXELS}"
echo "max_new_tokens=${MAX_NEW_TOKENS}"
echo "visual_enhancement=true"

EVAL_ARGS=(
  tools/evaluate_fewshot.py
  --model-path "${MODEL_PATH}"
  --data-root "${DATA_ROOT}"
  --work-dir "${WORK_ROOT}"
  --datasets "${DATASET_LIST[@]}"
  --shots "${SHOT_LIST[@]}"
  --device "${DEVICE}"
  --num-gpus "${NUM_GPUS}"
  --batch-size "${BATCH_SIZE}"
  --min-pixels "${MIN_PIXELS}"
  --max-pixels "${MAX_IMAGE_PIXELS}"
  --min-box-area-ratio "${MIN_BOX_AREA_RATIO}"
  --attention "${ATTENTION}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --ve
)
if [[ -n "${MAX_QUERY_PAIRS:-}" ]]; then
  EVAL_ARGS+=(--max-query-pairs "${MAX_QUERY_PAIRS}")
fi
if [[ "${SKIP_EXISTING:-0}" == "1" ]]; then
  EVAL_ARGS+=(--skip-existing)
fi
EVAL_ARGS+=("$@")

PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" "${PYTHON_BIN}" "${EVAL_ARGS[@]}"
