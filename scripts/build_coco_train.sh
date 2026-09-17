#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
COCO_ROOT="${COCO_ROOT:?Set COCO_ROOT to the COCO dataset directory}"
ANNOTATIONS="${ANNOTATIONS:-${COCO_ROOT}/annotations/instances_train2017.json}"
IMAGE_ROOT="${IMAGE_ROOT:-${COCO_ROOT}/train2017}"
OUTPUT="${OUTPUT:-${ROOT_DIR}/data/coco/train_sft.json}"
IMAGE_IDS_OUTPUT="${IMAGE_IDS_OUTPUT:-${ROOT_DIR}/data/coco/train_image_ids.json}"
SHOTS="${SHOTS:-1 2 4}"
NUM_SAMPLES="${NUM_SAMPLES:-11829}"
SEED="${SEED:-42}"
IMAGE_FRACTION="${IMAGE_FRACTION:-0.1}"

cd "${ROOT_DIR}"
read -r -a SHOT_LIST <<< "${SHOTS}"
"${PYTHON_BIN}" tools/build_coco.py train \
  --annotations "${ANNOTATIONS}" \
  --image-root "${IMAGE_ROOT}" \
  --output "${OUTPUT}" \
  --image-ids-output "${IMAGE_IDS_OUTPUT}" \
  --shots "${SHOT_LIST[@]}" \
  --num-samples "${NUM_SAMPLES}" \
  --image-fraction "${IMAGE_FRACTION}" \
  --seed "${SEED}"
