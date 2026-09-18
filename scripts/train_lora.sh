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
REPORT_TO="${REPORT_TO:-swanlab}"
RUN_NAME="${RUN_NAME:-$(basename "${OUTPUT_DIR}")}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-qwen3vl-coco-sft}"
SWANLAB_PROJ_NAME="${SWANLAB_PROJ_NAME:-${SWANLAB_PROJECT}}"
SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR:-${OUTPUT_DIR}/swanlog}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"

if [[ "${REPORT_TO}" == *swanlab* && -z "${SWANLAB_API_KEY:-}" ]]; then
  echo "SWANLAB_API_KEY must be set when REPORT_TO includes swanlab" >&2
  exit 1
fi
# The cluster's inherited proxy is unavailable on GPU nodes. Do not pass it to
# SwanLab or the training subprocess.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy
# SwanLab >=0.10 reads SWANLAB_PROJECT as a structured Settings value. Keep
# SWANLAB_PROJECT as the wrapper's public input, but expose the legacy scalar
# compatibility variable consumed by the SDK and do not leak the structured
# variable into the Trainer process.
unset SWANLAB_PROJECT
export SWANLAB_PROJ_NAME SWANLAB_LOG_DIR SWANLAB_MODE

cd "${ROOT_DIR}"
ARGS=(
  --model-name-or-path "${MODEL_NAME_OR_PATH}"
  --dataset "${DATASET}"
  --output-dir "${OUTPUT_DIR}"
  --run-name "${RUN_NAME}"
  --report-to "${REPORT_TO}"
  --eval-mode "${EVAL_MODE}"
  --eval-strategy "${EVAL_STRATEGY}"
  --lora-enable true
  --bf16 "${BF16:-true}"
  --fp16 "${FP16:-false}"
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}"
  --num-train-epochs "${NUM_TRAIN_EPOCHS:-12}"
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
  --per-device-eval-batch-size "${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-1}"
  --learning-rate "${LEARNING_RATE:-1e-4}"
  --model-max-length "${MODEL_MAX_LENGTH:-8192}"
  --min-pixels "${MIN_PIXELS:-3136}"
  --max-pixels "${MAX_PIXELS:-640000}"
  --save-strategy "${SAVE_STRATEGY:-epoch}"
  --save-total-limit "${SAVE_TOTAL_LIMIT:-2}"
  --save-steps "${SAVE_STEPS:-500}"
  --logging-steps "${LOGGING_STEPS:-10}"
)
if [[ -n "${EVAL_EPISODES}" ]]; then
  ARGS+=(
    --coco-eval-episodes "${EVAL_EPISODES}"
    --coco-eval-batch-size "${COCO_EVAL_BATCH_SIZE:-1}"
    --coco-eval-max-new-tokens "${COCO_EVAL_MAX_NEW_TOKENS:-1024}"
  )
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE}" \
  -m qwen3vl_sft.train "${ARGS[@]}"
