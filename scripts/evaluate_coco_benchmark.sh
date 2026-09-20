#!/usr/bin/env bash
set -euo pipefail

# Canonical COCO benchmark orchestration: build the fixed manifest if needed,
# then call the canonical fixed-episode evaluator.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH}"
EVAL_ID="${EVAL_ID:-coco-500}"
EVAL_ROOT="${EVAL_ROOT:-${ROOT_DIR}/outputs/eval/coco/${EVAL_ID}}"
EPISODES="${EPISODES:-${ROOT_DIR}/data/coco/val_episodes.json}"
OUTPUT="${OUTPUT:-${EVAL_ROOT}/result.json}"
SHOTS="${SHOTS:-0 1 2 4}"
NUM_QUERY_IMAGES="${NUM_QUERY_IMAGES:-500}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MIN_PIXELS="${MIN_PIXELS:-3136}"
MAX_PIXELS="${MAX_PIXELS:-640000}"
ATTENTION="${ATTENTION:-${ATTN_IMPLEMENTATION:-sdpa}}"
SUPPORT_IMAGE_IDS="${SUPPORT_IMAGE_IDS:-data/coco/train_image_ids.json}"
VE_MODE=0
VE_FLAG_IN_ARGS=0
IE_MODE=0
IE_FLAG_IN_ARGS=0
CATEGORY_DESCRIPTIONS_FLAG_IN_ARGS=0
if [[ "${VE:-0}" == "1" || "${VISUAL_ENHANCEMENT:-0}" == "1" ]]; then
  VE_MODE=1
fi
if [[ "${IE:-0}" == "1" || "${INSTRUCTION_ENHANCEMENT:-0}" == "1" ]]; then
  IE_MODE=1
fi
if [[ -n "${CATEGORY_DESCRIPTIONS:-}" ]]; then
  IE_MODE=1
fi
for argument in "$@"; do
  if [[ "${argument}" == "--ve" || "${argument}" == "--visual-enhancement" ]]; then
    VE_MODE=1
    VE_FLAG_IN_ARGS=1
  fi
  if [[ "${argument}" == "--ie" || "${argument}" == "--instruction-enhancement" ]]; then
    IE_MODE=1
    IE_FLAG_IN_ARGS=1
  fi
  if [[ "${argument}" == "--category-descriptions" || "${argument}" == --category-descriptions=* ]]; then
    IE_MODE=1
    CATEGORY_DESCRIPTIONS_FLAG_IN_ARGS=1
  fi
done
if [[ ! -f "${SUPPORT_IMAGE_IDS}" && -f data/coco/train_image_ids_10pct.json ]]; then
  SUPPORT_IMAGE_IDS=data/coco/train_image_ids_10pct.json
fi

if [[ ! -f "${EPISODES}" ]]; then
  : "${COCO_ROOT:?Set COCO_ROOT when ${EPISODES} does not exist}"
  if [[ ! -f "${SUPPORT_IMAGE_IDS}" ]]; then
    echo "Support image IDs not found: ${SUPPORT_IMAGE_IDS}" >&2
    echo "Set SUPPORT_IMAGE_IDS or build the COCO train manifest first." >&2
    exit 2
  fi
  mkdir -p "$(dirname "${EPISODES}")"
  COCO_ROOT="${COCO_ROOT}" \
    OUTPUT="${EPISODES}" \
    SUPPORT_IMAGE_IDS="${SUPPORT_IMAGE_IDS}" \
    SHOTS="${SHOTS}" \
    NUM_QUERY_IMAGES="${NUM_QUERY_IMAGES}" \
    bash scripts/build_coco_eval.sh
fi

mkdir -p "$(dirname "${OUTPUT}")"
LOG_FILE="${LOG_FILE:-${OUTPUT%.json}.log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "episodes=${EPISODES}"
echo "output=${OUTPUT}"
echo "num_query_images=${NUM_QUERY_IMAGES} when the manifest is built"
echo "max_new_tokens=${MAX_NEW_TOKENS}"
echo "visual_enhancement=${VE_MODE}"
echo "instruction_enhancement=${IE_MODE}"
if [[ -n "${CATEGORY_DESCRIPTIONS:-}" ]]; then
  echo "category_descriptions=${CATEGORY_DESCRIPTIONS}"
fi

EVAL_ARGS=(
  tools/evaluate_coco.py
  --model-path "${MODEL_NAME_OR_PATH}"
  --episodes "${EPISODES}"
  --output "${OUTPUT}"
  --device "${DEVICE}"
  --batch-size "${BATCH_SIZE}"
  --min-pixels "${MIN_PIXELS}"
  --max-pixels "${MAX_PIXELS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --attention "${ATTENTION}"
)
if [[ -n "${ADAPTER_PATH:-}" ]]; then
  EVAL_ARGS+=(--adapter-path "${ADAPTER_PATH}")
fi
if [[ "${VE_MODE}" == "1" && "${VE_FLAG_IN_ARGS}" == "0" ]]; then
  EVAL_ARGS+=(--ve)
fi
if [[ "${IE_MODE}" == "1" && "${IE_FLAG_IN_ARGS}" == "0" ]]; then
  EVAL_ARGS+=(--instruction-enhancement)
fi
if [[ -n "${CATEGORY_DESCRIPTIONS:-}" && "${CATEGORY_DESCRIPTIONS_FLAG_IN_ARGS}" == "0" ]]; then
  EVAL_ARGS+=(--category-descriptions "${CATEGORY_DESCRIPTIONS}")
fi
EVAL_ARGS+=("$@")

PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" "${PYTHON_BIN}" "${EVAL_ARGS[@]}"
