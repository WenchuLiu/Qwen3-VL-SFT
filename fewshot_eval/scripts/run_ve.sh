#!/usr/bin/env bash
set -euo pipefail

# Visual Enhancement few-shot evaluation for Qwen3-VL-4B.
# VE draws red GT boxes on support images only; query images remain untouched.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-${CKPT:-${ROOT_DIR}/weights/Qwen3-VL-4B-Instruct}}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data}"
WORK_ROOT="${WORK_ROOT:-${ROOT_DIR}/work_dirs/qwen3-vl-4b-ve-fewshot}"
DATASETS="${DATASETS:-ArTaxOr Clipart1k FISH NEU-DET UODD}"
SHOTS="${SHOTS:-1 2 4}"
DEVICE="${DEVICE:-cuda:0}"
NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MIN_PIXELS="${MIN_PIXELS:-3136}"
MAX_PIXELS="${MAX_PIXELS:-640000}"
MIN_BOX_AREA_RATIO="${MIN_BOX_AREA_RATIO:-0}"
ATTENTION="${ATTENTION:-sdpa}"

cd "${ROOT_DIR}"
read -r -a DATASET_LIST <<< "${DATASETS}"
read -r -a SHOT_LIST <<< "${SHOTS}"
mkdir -p "${WORK_ROOT}"
LOG_FILE="${LOG_FILE:-${WORK_ROOT}/evaluation.log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "model_path=${MODEL_PATH}"
echo "data_root=${DATA_ROOT}"
echo "work_root=${WORK_ROOT}"
echo "datasets=${DATASETS}"
echo "shots=${SHOTS}"
echo "num_gpus=${NUM_GPUS}"
echo "batch_size=${BATCH_SIZE}"
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
  --max-pixels "${MAX_PIXELS}"
  --min-box-area-ratio "${MIN_BOX_AREA_RATIO}"
  --attention "${ATTENTION}"
  --ve
)
if [[ -n "${MAX_NEW_TOKENS:-}" ]]; then
  EVAL_ARGS+=(--max-new-tokens "${MAX_NEW_TOKENS}")
fi
if [[ -n "${MAX_QUERY_PAIRS:-}" ]]; then
  EVAL_ARGS+=(--max-query-pairs "${MAX_QUERY_PAIRS}")
fi
if [[ "${SKIP_EXISTING:-0}" == "1" ]]; then
  EVAL_ARGS+=(--skip-existing)
fi
EVAL_ARGS+=("$@")

PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" "${PYTHON_BIN}" "${EVAL_ARGS[@]}"
