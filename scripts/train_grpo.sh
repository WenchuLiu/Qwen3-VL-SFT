#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH to a model path or ID}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
DATASET="${DATASET:?Set DATASET to a JSON/JSONL GRPO file}"
DATA_ROOT="${DATA_ROOT:-}"
RUN_ID="${RUN_ID:-qwen3vl-grpo-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/grpo/${RUN_ID}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
REPORT_TO="${REPORT_TO:-none}"
RUN_NAME="${RUN_NAME:-$(basename "${OUTPUT_DIR}")}"
REWARD_FUNCTIONS="${REWARD_FUNCTIONS:-iou score format}"
read -r -a REWARD_LIST <<< "${REWARD_FUNCTIONS}"

if [[ "${REPORT_TO}" == *swanlab* && -z "${SWANLAB_API_KEY:-}" ]]; then
  echo "SWANLAB_API_KEY must be set when REPORT_TO includes swanlab" >&2
  exit 1
fi

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy
cd "${ROOT_DIR}"

ARGS=(
  --model-name-or-path "${MODEL_NAME_OR_PATH}"
  --dataset "${DATASET}"
  --output-dir "${OUTPUT_DIR}"
  --run-name "${RUN_NAME}"
  --report-to "${REPORT_TO}"
  --eval-mode none
  --eval-strategy no
  --reward-functions "${REWARD_LIST[@]}"
  --num-generations "${NUM_GENERATIONS:-4}"
  --max-prompt-length "${MAX_PROMPT_LENGTH:-4096}"
  --max-completion-length "${MAX_COMPLETION_LENGTH:-512}"
  --kl-coef "${KL_COEF:-0.04}"
  --iou-threshold "${IOU_THRESHOLD:-0.5}"
  --iou-reward-weight "${IOU_REWARD_WEIGHT:-1.0}"
  --score-reward-weight "${SCORE_REWARD_WEIGHT:-1.0}"
  --format-reward-weight "${FORMAT_REWARD_WEIGHT:-1.0}"
  --lora-enable "${LORA_ENABLE:-true}"
  --bf16 "${BF16:-true}"
  --fp16 "${FP16:-false}"
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}"
  --num-train-epochs "${NUM_TRAIN_EPOCHS:-2}"
  --max-steps "${MAX_STEPS:--1}"
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-1}"
  --learning-rate "${LEARNING_RATE:-1e-5}"
  --model-max-length "${MODEL_MAX_LENGTH:-8192}"
  --min-pixels "${MIN_PIXELS:-3136}"
  --max-pixels "${MAX_PIXELS:-640000}"
  --save-strategy "${SAVE_STRATEGY:-epoch}"
  --save-total-limit "${SAVE_TOTAL_LIMIT:-2}"
  --logging-steps "${LOGGING_STEPS:-1}"
)
if [[ -n "${REFERENCE_MODEL_NAME_OR_PATH:-}" ]]; then
  ARGS+=(--reference-model-name-or-path "${REFERENCE_MODEL_NAME_OR_PATH}")
fi
if [[ -n "${ADAPTER_PATH}" ]]; then
  ARGS+=(--adapter-path "${ADAPTER_PATH}")
fi
if [[ -n "${DATA_ROOT}" ]]; then
  ARGS+=(--data-root "${DATA_ROOT}")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE}" \
  -m qwen3vl_sft.train.grpo "${ARGS[@]}"
