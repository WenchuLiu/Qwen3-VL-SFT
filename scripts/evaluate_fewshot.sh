#!/usr/bin/env bash
set -euo pipefail

# Canonical launcher for the cross-domain few-shot benchmark.
# Set VE=1 for Visual Enhancement, IE=1 for Instruction Enhancement, or
# DETPO=1 for the per-shot DetPO prompt files.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-${CKPT:-weights/Qwen3-VL-4B-Instruct}}"
DATA_ROOT="${DATA_ROOT:-data}"
DEFAULT_DATASETS=(ArTaxOr clipart1k FISH NEU-DET UODD VISUALDIOR)
VE_MODE=0
VE_FLAG_IN_ARGS=0
IE_MODE=0
IE_FLAG_IN_ARGS=0
CATEGORY_DESCRIPTIONS_FLAG_IN_ARGS=0
DETPO_MODE=0
DETPO_FLAG_IN_ARGS=0
DETPO_PROMPTS_FLAG_IN_ARGS=0
if [[ "${VE:-0}" == "1" || "${VISUAL_ENHANCEMENT:-0}" == "1" ]]; then
  VE_MODE=1
fi
if [[ "${IE:-0}" == "1" || "${INSTRUCTION_ENHANCEMENT:-0}" == "1" ]]; then
  IE_MODE=1
fi
if [[ "${DETPO:-0}" == "1" || "${DETPO_PROMPT:-0}" == "1" ]]; then
  DETPO_MODE=1
fi
if [[ -n "${DETPO_PROMPTS:-}" ]]; then
  DETPO_MODE=1
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
    CATEGORY_DESCRIPTIONS_FLAG_IN_ARGS=1
  fi
  if [[ "${argument}" == "--detpo" ]]; then
    DETPO_MODE=1
    DETPO_FLAG_IN_ARGS=1
  fi
  if [[ "${argument}" == "--detpo-prompts" || "${argument}" == --detpo-prompts=* || "${argument}" == "--detpo-prompt-root" || "${argument}" == --detpo-prompt-root=* ]]; then
    DETPO_MODE=1
    DETPO_PROMPTS_FLAG_IN_ARGS=1
  fi
done

# A description file supplies DetPO's 0-shot prompt too; infer IE only when
# DetPO has not been selected. Explicit IE + DetPO remains an error.
if [[ "${DETPO_MODE}" == "0" && ( -n "${CATEGORY_DESCRIPTIONS:-}" || "${CATEGORY_DESCRIPTIONS_FLAG_IN_ARGS}" == "1" ) ]]; then
  IE_MODE=1
fi

if [[ "${DETPO_MODE}" == "1" && "${IE_MODE}" == "1" ]]; then
  echo "DETPO and IE are mutually exclusive prompt modes; choose one." >&2
  exit 2
fi
if [[ "${DETPO_MODE}" == "1" && "${VE_MODE}" == "1" ]]; then
  echo "DetPO uses the original single-image prompt and cannot be combined with VE." >&2
  exit 2
fi

if [[ "${VE_MODE}" == "1" ]]; then
  DEFAULT_SHOTS=(1 2 4)
  if [[ "${DETPO_MODE}" == "1" ]]; then
    EVAL_ID="${EVAL_ID:-qwen3-vl-4b-ve-detpo-fewshot}"
  elif [[ "${IE_MODE}" == "1" ]]; then
    EVAL_ID="${EVAL_ID:-qwen3-vl-4b-ve-ie-fewshot}"
  else
    EVAL_ID="${EVAL_ID:-qwen3-vl-4b-ve-fewshot}"
  fi
  DEFAULT_NUM_GPUS=4
  DEFAULT_BATCH_SIZE=2
elif [[ "${DETPO_MODE}" == "1" ]]; then
  DEFAULT_SHOTS=(1 2 4)
  EVAL_ID="${EVAL_ID:-qwen3-vl-4b-detpo-fewshot}"
  DEFAULT_NUM_GPUS=2
  DEFAULT_BATCH_SIZE=1
elif [[ "${IE_MODE}" == "1" ]]; then
  DEFAULT_SHOTS=(0 1 2 4)
  EVAL_ID="${EVAL_ID:-qwen3-vl-4b-ie-fewshot}"
  DEFAULT_NUM_GPUS=2
  DEFAULT_BATCH_SIZE=1
else
  DEFAULT_SHOTS=(0 1 2 4)
  EVAL_ID="${EVAL_ID:-qwen3-vl-4b-fewshot}"
  DEFAULT_NUM_GPUS=2
  DEFAULT_BATCH_SIZE=1
fi
DEFAULT_WORK_ROOT="${ROOT_DIR}/outputs/eval/fewshot/${EVAL_ID}"

if [[ -n "${DATASETS:-}" ]]; then
  read -r -a DATASET_LIST <<< "${DATASETS}"
else
  DATASET_LIST=("${DEFAULT_DATASETS[@]}")
fi
if [[ -n "${SHOTS:-}" ]]; then
  read -r -a SHOT_LIST <<< "${SHOTS}"
else
  SHOT_LIST=("${DEFAULT_SHOTS[@]}")
fi

WORK_ROOT="${WORK_ROOT:-${OUTPUT_DIR:-${EVAL_ROOT:-${DEFAULT_WORK_ROOT}}}}"
NUM_GPUS="${NUM_GPUS:-${DEFAULT_NUM_GPUS}}"
BATCH_SIZE="${BATCH_SIZE:-${DEFAULT_BATCH_SIZE}}"
MIN_PIXELS="${MIN_PIXELS:-3136}"
MAX_IMAGE_PIXELS="${MAX_IMAGE_PIXELS:-${MAX_PIXELS:-640000}}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
MIN_BOX_AREA_RATIO="${MIN_BOX_AREA_RATIO:-0}"
ATTENTION="${ATTENTION:-sdpa}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-43}"
DETPO_PROMPTS="${DETPO_PROMPTS:-${ROOT_DIR}/docs/cross-domain-instructions}"

mkdir -p "${WORK_ROOT}"
LOG_FILE="${LOG_FILE:-${WORK_ROOT}/evaluation.log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "model_path=${MODEL_PATH}"
echo "data_root=${DATA_ROOT}"
echo "work_root=${WORK_ROOT}"
echo "datasets=${DATASET_LIST[*]}"
echo "shots=${SHOT_LIST[*]}"
echo "num_gpus=${NUM_GPUS}"
echo "batch_size=${BATCH_SIZE}"
echo "max_image_pixels=${MAX_IMAGE_PIXELS}"
echo "attention=${ATTENTION}"
echo "seed=${SEED}"
if [[ -n "${MAX_NEW_TOKENS}" ]]; then
  echo "max_new_tokens_override=${MAX_NEW_TOKENS}"
else
  echo "max_new_tokens_policy=dataset-default (VISUALDIOR=2048, others=1024)"
fi
echo "visual_enhancement=${VE_MODE}"
echo "instruction_enhancement=${IE_MODE}"
echo "detpo=${DETPO_MODE}"
if [[ "${DETPO_MODE}" == "1" ]]; then
  echo "detpo_prompts=${DETPO_PROMPTS}"
fi
if [[ -n "${CATEGORY_DESCRIPTIONS:-}" ]]; then
  echo "category_descriptions=${CATEGORY_DESCRIPTIONS}"
fi

EVAL_ARGS=(
  tools/evaluate_fewshot.py
  --model-path "${MODEL_PATH}"
  --data-root "${DATA_ROOT}"
  --work-dir "${WORK_ROOT}"
  --datasets "${DATASET_LIST[@]}"
  --shots "${SHOT_LIST[@]}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --num-gpus "${NUM_GPUS}"
  --batch-size "${BATCH_SIZE}"
  --min-pixels "${MIN_PIXELS}"
  --max-pixels "${MAX_IMAGE_PIXELS}"
  --min-box-area-ratio "${MIN_BOX_AREA_RATIO}"
  --attention "${ATTENTION}"
)
if [[ -n "${MAX_NEW_TOKENS}" ]]; then
  EVAL_ARGS+=(--max-new-tokens "${MAX_NEW_TOKENS}")
fi
if [[ "${VE_MODE}" == "1" && "${VE_FLAG_IN_ARGS}" == "0" ]]; then
  EVAL_ARGS+=(--ve)
fi
if [[ "${IE_MODE}" == "1" && "${IE_FLAG_IN_ARGS}" == "0" ]]; then
  EVAL_ARGS+=(--instruction-enhancement)
fi
if [[ "${DETPO_MODE}" == "1" && "${DETPO_FLAG_IN_ARGS}" == "0" ]]; then
  EVAL_ARGS+=(--detpo)
fi
if [[ "${DETPO_MODE}" == "1" && "${DETPO_PROMPTS_FLAG_IN_ARGS}" == "0" ]]; then
  EVAL_ARGS+=(--detpo-prompts "${DETPO_PROMPTS}")
fi
if [[ -n "${CATEGORY_DESCRIPTIONS:-}" && "${CATEGORY_DESCRIPTIONS_FLAG_IN_ARGS}" == "0" ]]; then
  EVAL_ARGS+=(--category-descriptions "${CATEGORY_DESCRIPTIONS}")
fi
if [[ -n "${MAX_QUERY_PAIRS:-}" ]]; then
  EVAL_ARGS+=(--max-query-pairs "${MAX_QUERY_PAIRS}")
fi
if [[ "${SKIP_EXISTING:-0}" == "1" ]]; then
  EVAL_ARGS+=(--skip-existing)
fi
if [[ "${FLAT_WORK_DIR:-0}" == "1" ]]; then
  EVAL_ARGS+=(--flat-work-dir)
fi
EVAL_ARGS+=("$@")

PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" "${PYTHON_BIN}" "${EVAL_ARGS[@]}"
