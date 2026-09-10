#!/usr/bin/env bash
set -euo pipefail

# Reproducible protocol: build once, evaluate the base model, train, then
# evaluate the adapter on the exact same fixed episode file.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${ROOT_DIR}/runs/coco-before-after}"
TRAIN_DATA="${TRAIN_DATA:-${EXPERIMENT_DIR}/train.json}"
IMAGE_IDS="${IMAGE_IDS:-${EXPERIMENT_DIR}/train_image_ids.json}"
EPISODES="${EPISODES:-${EXPERIMENT_DIR}/eval_episodes.json}"
BASE_RESULT="${BASE_RESULT:-${EXPERIMENT_DIR}/base.json}"
ADAPTER_RESULT="${ADAPTER_RESULT:-${EXPERIMENT_DIR}/adapter.json}"

mkdir -p "${EXPERIMENT_DIR}"
COCO_ROOT="${COCO_ROOT:?Set COCO_ROOT}" \
  OUTPUT="${TRAIN_DATA}" IMAGE_IDS_OUTPUT="${IMAGE_IDS}" \
  bash "${ROOT_DIR}/scripts/build_coco_train.sh"
COCO_ROOT="${COCO_ROOT}" SUPPORT_IMAGE_IDS="${IMAGE_IDS}" OUTPUT="${EPISODES}" \
  bash "${ROOT_DIR}/scripts/build_coco_eval.sh"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH}" \
  EPISODES="${EPISODES}" OUTPUT="${BASE_RESULT}" \
  bash "${ROOT_DIR}/scripts/evaluate_coco.sh"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}" DATASET="${TRAIN_DATA}" \
  OUTPUT_DIR="${EXPERIMENT_DIR}/adapter" EVAL_MODE=generation \
  EVAL_STRATEGY=epoch EVAL_EPISODES="${EPISODES}" \
  bash "${ROOT_DIR}/scripts/train_lora.sh"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}" ADAPTER_PATH="${EXPERIMENT_DIR}/adapter" \
  EPISODES="${EPISODES}" OUTPUT="${ADAPTER_RESULT}" \
  bash "${ROOT_DIR}/scripts/evaluate_coco.sh"
"${PYTHON_BIN}" "${ROOT_DIR}/tools/compare.py" \
  --before "${BASE_RESULT}" --after "${ADAPTER_RESULT}" \
  --output "${EXPERIMENT_DIR}/comparison.json"
