#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# The local symlinks created for this workspace make these defaults resolve to
# /data/Qwen3-VL/weights/Qwen3-VL-4B-Instruct and /data/LLM respectively.
CKPT="${CKPT:-${ROOT_DIR}/weights/Qwen3-VL-4B-Instruct}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data}"
WORK_DIR="${WORK_DIR:-${ROOT_DIR}/work_dirs/qwen3-vl-4b-base-fewshot}"
DATASETS="${DATASETS:-ArTaxOr Clipart1k FISH NEU-DET UODD}"
SHOTS="${SHOTS:-0 1 2 4}"
NUM_GPUS="${NUM_GPUS:-2}"

mkdir -p "${WORK_DIR}"
cd "${ROOT_DIR}"

# Keep a single mmdetection-style run log beside config.json, summary.json,
# and the per-dataset/per-shot result files.
exec > >(tee -a "${WORK_DIR}/evaluation.log") 2>&1

read -r -a DATASET_LIST <<< "${DATASETS}"
read -r -a SHOT_LIST <<< "${SHOTS}"
OPTIONAL_ARGS=()
if [[ -n "${MAX_QUERY_PAIRS:-}" ]]; then
  OPTIONAL_ARGS+=(--max-query-pairs "${MAX_QUERY_PAIRS}")
fi

echo "checkpoint=${CKPT}"
echo "data_root=${DATA_ROOT}"
echo "work_dir=${WORK_DIR}"
echo "datasets=${DATASETS}"
echo "shots=${SHOTS}"
echo "num_gpus=${NUM_GPUS}"

PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" "${PYTHON_BIN}" tools/evaluate_fewshot.py \
  --model-path "${CKPT}" \
  --data-root "${DATA_ROOT}" \
  --work-dir "${WORK_DIR}" \
  --datasets "${DATASET_LIST[@]}" \
  --shots "${SHOT_LIST[@]}" \
  --device "${DEVICE:-cuda:0}" \
  --num-gpus "${NUM_GPUS}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --min-pixels "${MIN_PIXELS:-3136}" \
  --max-pixels "${MAX_PIXELS:-640000}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-256}" \
  --min-box-area-ratio "${MIN_BOX_AREA_RATIO:-0}" \
  --attention "${ATTENTION:-sdpa}" \
  "${OPTIONAL_ARGS[@]}" \
  ${SKIP_EXISTING:+--skip-existing}
