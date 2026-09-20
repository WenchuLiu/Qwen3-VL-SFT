#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH}"
EPISODES="${EPISODES:?Set EPISODES to a fixed episode JSON file}"
EVAL_ID="${EVAL_ID:-coco-eval}"
EVAL_ROOT="${EVAL_ROOT:-${ROOT_DIR}/outputs/eval/coco/${EVAL_ID}}"
OUTPUT="${OUTPUT:-${EVAL_ROOT}/result.json}"

cd "${ROOT_DIR}"
ADAPTER_ARGS=()
if [[ -n "${ADAPTER_PATH:-}" ]]; then
  ADAPTER_ARGS+=(--adapter-path "${ADAPTER_PATH}")
fi
VE_ARGS=()
if [[ "${VE:-0}" == "1" || "${VISUAL_ENHANCEMENT:-0}" == "1" ]]; then
  VE_ARGS+=(--ve)
fi
PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" "${PYTHON_BIN}" tools/evaluate_coco.py \
  --model-path "${MODEL_NAME_OR_PATH}" \
  "${ADAPTER_ARGS[@]}" \
  "${VE_ARGS[@]}" \
  --episodes "${EPISODES}" \
  --output "${OUTPUT}" \
  --device "${DEVICE:-cuda:0}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --min-pixels "${MIN_PIXELS:-3136}" \
  --max-pixels "${MAX_PIXELS:-640000}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-1024}" \
  --attention "${ATTN_IMPLEMENTATION:-sdpa}"
