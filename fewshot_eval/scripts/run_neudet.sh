#!/usr/bin/env bash
set -euo pipefail

# Compatibility launcher. Use scripts/evaluate_fewshot.sh for new runs.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
if [[ -n "${DATASET_WORK_DIR:-}" ]]; then
  WORK_ROOT="${DATASET_WORK_DIR}"
  FLAT_WORK_DIR=1
else
  WORK_ROOT="${WORK_ROOT:-${WORK_DIR:-work_dirs/qwen3-vl-4b-fewshot}}"
  FLAT_WORK_DIR="${FLAT_WORK_DIR:-0}"
fi
DATASETS=NEU-DET SHOTS="${SHOTS:-0 1 2 4}" \
  WORK_ROOT="${WORK_ROOT}" FLAT_WORK_DIR="${FLAT_WORK_DIR}" \
  bash scripts/evaluate_fewshot.sh "$@"
