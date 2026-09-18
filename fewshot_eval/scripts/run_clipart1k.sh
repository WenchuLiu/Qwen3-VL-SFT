#!/usr/bin/env bash
set -euo pipefail

# Clipart1k: 1024 generated tokens, matching ../Qwen3-VL.
DATASET="Clipart1k"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-${CKPT:-${ROOT_DIR}/weights/Qwen3-VL-4B-Instruct}}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data}"
WORK_ROOT="${WORK_ROOT:-${WORK_DIR:-${ROOT_DIR}/work_dirs/qwen3-vl-4b-base-fewshot}}"
DATASET_WORK_DIR="${DATASET_WORK_DIR:-${WORK_ROOT}/${DATASET}}"
SHOTS="${SHOTS:-0 1 2 4}"
DEVICE="${DEVICE:-cuda:0}"
NUM_GPUS="${NUM_GPUS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MIN_PIXELS="${MIN_PIXELS:-3136}"
MAX_PIXELS="${MAX_PIXELS:-640000}"
MIN_BOX_AREA_RATIO="${MIN_BOX_AREA_RATIO:-0}"
ATTENTION="${ATTENTION:-sdpa}"

cd "${ROOT_DIR}"
read -r -a SHOT_LIST <<< "${SHOTS}"
mkdir -p "${DATASET_WORK_DIR}"
LOG_FILE="${LOG_FILE:-${DATASET_WORK_DIR}/evaluation.log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "dataset=${DATASET}"
echo "model_path=${MODEL_PATH}"
echo "data_root=${DATA_ROOT}"
echo "work_dir=${DATASET_WORK_DIR}"
echo "shots=${SHOTS}"
echo "num_gpus=${NUM_GPUS}"
echo "max_new_tokens=${MAX_NEW_TOKENS}"

EVAL_ARGS=(
  tools/evaluate_fewshot.py
  --model-path "${MODEL_PATH}"
  --data-root "${DATA_ROOT}"
  --work-dir "${DATASET_WORK_DIR}"
  --datasets "${DATASET}"
  --shots "${SHOT_LIST[@]}"
  --device "${DEVICE}"
  --num-gpus "${NUM_GPUS}"
  --batch-size "${BATCH_SIZE}"
  --min-pixels "${MIN_PIXELS}"
  --max-pixels "${MAX_PIXELS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --flat-work-dir
  --min-box-area-ratio "${MIN_BOX_AREA_RATIO}"
  --attention "${ATTENTION}"
)
if [[ -n "${MAX_QUERY_PAIRS:-}" ]]; then
  EVAL_ARGS+=(--max-query-pairs "${MAX_QUERY_PAIRS}")
fi
if [[ "${SKIP_EXISTING:-0}" == "1" ]]; then
  EVAL_ARGS+=(--skip-existing)
fi
EVAL_ARGS+=("$@")

set +e
PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" "${PYTHON_BIN}" "${EVAL_ARGS[@]}"
EXIT_CODE=$?
set -e

echo "dataset=${DATASET} exit_code=${EXIT_CODE}"
exit "${EXIT_CODE}"
