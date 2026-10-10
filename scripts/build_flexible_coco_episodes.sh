#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
COCO_ROOT="${COCO_ROOT:-${ROOT_DIR}/data/COCO}"
ANNOTATIONS="${ANNOTATIONS:-${COCO_ROOT}/annotations/instances_train2017.json}"
IMAGE_ROOT="${IMAGE_ROOT:-${COCO_ROOT}/train2017}"
IMAGE_IDS="${IMAGE_IDS:-${ROOT_DIR}/docs/coco_train_image_ids_10pct_seed42.json}"
UPPER_CATEGORY_MAP="${UPPER_CATEGORY_MAP:-${ROOT_DIR}/docs/coco_upper_category_map.json}"
OUTPUT="${OUTPUT:-${ROOT_DIR}/data/coco/train_episodes_10pct_flexible_11820.json}"
NUM_EPISODES="${NUM_EPISODES:-11820}"
SEED="${SEED:-42}"
IMAGE_POOL_SEED="${IMAGE_POOL_SEED:-42}"

cd "${ROOT_DIR}"
if [[ ! -f "${IMAGE_IDS}" ]]; then
  mkdir -p "$(dirname "${IMAGE_IDS}")"
  "${PYTHON_BIN}" tools/build_coco.py image-ids \
    --annotations "${ANNOTATIONS}" \
    --output "${IMAGE_IDS}" \
    --fraction 0.1 \
    --seed "${IMAGE_POOL_SEED}"
fi

"${PYTHON_BIN}" tools/build_coco.py episodes \
  --annotations "${ANNOTATIONS}" \
  --image-root "${IMAGE_ROOT}" \
  --image-ids "${IMAGE_IDS}" \
  --image-pool-seed "${IMAGE_POOL_SEED}" \
  --upper-category-map "${UPPER_CATEGORY_MAP}" \
  --output "${OUTPUT}" \
  --num-episodes "${NUM_EPISODES}" \
  --seed "${SEED}"
