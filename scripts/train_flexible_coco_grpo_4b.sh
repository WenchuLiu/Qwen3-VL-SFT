#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
COCO_ROOT="${COCO_ROOT:-${ROOT_DIR}/data/COCO}"
ANNOTATIONS="${ANNOTATIONS:-${COCO_ROOT}/annotations/instances_train2017.json}"
IMAGE_ROOT="${IMAGE_ROOT:-${COCO_ROOT}/train2017}"
IMAGE_IDS="${IMAGE_IDS:-${ROOT_DIR}/docs/coco_train_image_ids_10pct_seed42.json}"
UPPER_CATEGORY_MAP="${UPPER_CATEGORY_MAP:-${ROOT_DIR}/docs/coco_upper_category_map.json}"
DATASET="${DATASET:-${ROOT_DIR}/data/coco/train_episodes_10pct_flexible_11820.json}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${ROOT_DIR}/weights/Qwen3-VL-4B-Instruct}"
RUN_ID="${RUN_ID:-flexible-coco-grpo-4b-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/grpo/${RUN_ID}}"
NUM_EPISODES="${NUM_EPISODES:-11820}"
IMAGE_POOL_SEED="${IMAGE_POOL_SEED:-42}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

cd "${ROOT_DIR}"

# This run must start from the original checkpoint already present on disk.
# Validate it before building data so a missing path cannot trigger a Hub lookup.
if [[ ! -d "${MODEL_NAME_OR_PATH}" ]]; then
  echo "local model directory not found: ${MODEL_NAME_OR_PATH}" >&2
  echo "set MODEL_NAME_OR_PATH to the local Qwen3-VL-4B-Instruct directory" >&2
  exit 1
fi
MODEL_NAME_OR_PATH="$(cd "${MODEL_NAME_OR_PATH}" && pwd)"
if [[ ! -f "${MODEL_NAME_OR_PATH}/config.json" ]]; then
  echo "model config not found: ${MODEL_NAME_OR_PATH}/config.json" >&2
  exit 1
fi

# Keep all model and dataset loader calls offline for this local-checkpoint run.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

case "${BUILD_DATASET:-true}" in
  1|true|TRUE|yes|YES)
    COCO_ROOT="${COCO_ROOT}" \
    ANNOTATIONS="${ANNOTATIONS}" \
    IMAGE_ROOT="${IMAGE_ROOT}" \
    IMAGE_IDS="${IMAGE_IDS}" \
    UPPER_CATEGORY_MAP="${UPPER_CATEGORY_MAP}" \
    OUTPUT="${DATASET}" \
    NUM_EPISODES="${NUM_EPISODES}" \
    IMAGE_POOL_SEED="${IMAGE_POOL_SEED}" \
    SEED="${SEED:-42}" \
    PYTHON_BIN="${PYTHON_BIN}" \
      bash scripts/build_flexible_coco_episodes.sh
    ;;
  0|false|FALSE|no|NO)
    if [[ ! -f "${DATASET}" ]]; then
      echo "episode dataset not found: ${DATASET}" >&2
      echo "set BUILD_DATASET=true or build it with scripts/build_flexible_coco_episodes.sh" >&2
      exit 1
    fi
    ;;
  *)
    echo "BUILD_DATASET must be true or false" >&2
    exit 1
    ;;
esac

# This run starts from the original 4B checkpoint and creates a fresh LoRA.
# Do not inherit an SFT adapter or an external frozen reference model.
ADAPTER_PATH="" \
REFERENCE_MODEL_NAME_OR_PATH="" \
LORA_ENABLE=true \
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}" \
DATASET="${DATASET}" \
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
RUN_ID="${RUN_ID}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-${NPROC_PER_NODE}}" \
ROLLOUT_N="${ROLLOUT_N:-4}" \
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}" \
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}" \
MIN_PIXELS="${MIN_PIXELS:-3136}" \
MAX_PIXELS="${MAX_PIXELS:-640000}" \
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}" \
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}" \
LEARNING_RATE="${LEARNING_RATE:-1e-5}" \
BF16="${BF16:-true}" \
FP16="${FP16:-false}" \
REPORT_TO="${REPORT_TO:-none}" \
PYTHON_BIN="${PYTHON_BIN}" \
  bash scripts/train_grpo.sh
