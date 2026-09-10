#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH to a local path or Hugging Face model ID}"
DATASET="${DATASET:?Set DATASET to a JSON/JSONL SFT file}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/runs/qwen3vl-lora}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
EVAL_MODE="${EVAL_MODE:-none}"
EVAL_STRATEGY="${EVAL_STRATEGY:-no}"
EVAL_EPISODES="${EVAL_EPISODES:-}"

cd "${ROOT_DIR}"
ARGS=(
  --model-name-or-path "${MODEL_NAME_OR_PATH}"
  --dataset "${DATASET}"
  --output-dir "${OUTPUT_DIR}"
  --eval-mode "${EVAL_MODE}"
  --eval-strategy "${EVAL_STRATEGY}"
  --lora-enable true
  --bf16 "${BF16:-true}"
  --fp16 "${FP16:-false}"
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}"
  --num-train-epochs "${NUM_TRAIN_EPOCHS:-1}"
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
  --per-device-eval-batch-size "${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-16}"
  --learning-rate "${LEARNING_RATE:-2e-4}"
  --model-max-length "${MODEL_MAX_LENGTH:-8192}"
  --min-pixels "${MIN_PIXELS:-3136}"
  --max-pixels "${MAX_PIXELS:-640000}"
  --save-steps "${SAVE_STEPS:-500}"
  --logging-steps "${LOGGING_STEPS:-10}"
)
if [[ -n "${EVAL_EPISODES}" ]]; then
  ARGS+=(
    --coco-eval-episodes "${EVAL_EPISODES}"
    --coco-eval-batch-size "${COCO_EVAL_BATCH_SIZE:-1}"
    --coco-eval-max-new-tokens "${COCO_EVAL_MAX_NEW_TOKENS:-256}"
  )
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE}" \
  -m qwen3vl_sft.train "${ARGS[@]}"
