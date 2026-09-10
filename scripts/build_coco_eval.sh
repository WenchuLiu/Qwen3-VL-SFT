#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
COCO_ROOT="${COCO_ROOT:?Set COCO_ROOT to the COCO dataset directory}"
SUPPORT_ANNOTATIONS="${SUPPORT_ANNOTATIONS:-${COCO_ROOT}/annotations/instances_train2017.json}"
SUPPORT_IMAGE_ROOT="${SUPPORT_IMAGE_ROOT:-${COCO_ROOT}/train2017}"
SUPPORT_IMAGE_IDS="${SUPPORT_IMAGE_IDS:-${ROOT_DIR}/data/coco/train_image_ids.json}"
QUERY_ANNOTATIONS="${QUERY_ANNOTATIONS:-${COCO_ROOT}/annotations/instances_val2017.json}"
QUERY_IMAGE_ROOT="${QUERY_IMAGE_ROOT:-${COCO_ROOT}/val2017}"
OUTPUT="${OUTPUT:-${ROOT_DIR}/data/coco/val_episodes.json}"
SHOTS="${SHOTS:-1}"
NUM_QUERY_IMAGES="${NUM_QUERY_IMAGES:-500}"
SEED="${SEED:-43}"

cd "${ROOT_DIR}"
read -r -a SHOT_LIST <<< "${SHOTS}"
"${PYTHON_BIN}" tools/build_coco.py eval \
  --support-annotations "${SUPPORT_ANNOTATIONS}" \
  --support-image-root "${SUPPORT_IMAGE_ROOT}" \
  --support-image-ids "${SUPPORT_IMAGE_IDS}" \
  --query-annotations "${QUERY_ANNOTATIONS}" \
  --query-image-root "${QUERY_IMAGE_ROOT}" \
  --output "${OUTPUT}" \
  --shots "${SHOT_LIST[@]}" \
  --num-query-images "${NUM_QUERY_IMAGES}" \
  --seed "${SEED}"
