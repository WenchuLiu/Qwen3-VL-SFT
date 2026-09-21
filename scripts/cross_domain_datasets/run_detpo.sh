#!/usr/bin/env bash
set -euo pipefail

# Compatibility launcher for DetPO prompt evaluation.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DETPO=1 \
  WORK_ROOT="${WORK_ROOT:-${OUTPUT_DIR:-${ROOT_DIR}/outputs/eval/fewshot/qwen3-vl-4b-detpo-fewshot}}" \
  exec bash scripts/evaluate_fewshot.sh "$@"
