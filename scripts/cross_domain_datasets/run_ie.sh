#!/usr/bin/env bash
set -euo pipefail

# Compatibility launcher for instruction-enhanced few-shot evaluation.
# New experiments can use scripts/evaluate_fewshot.sh with IE=1 directly.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
IE=1 WORK_ROOT="${WORK_ROOT:-${OUTPUT_DIR:-${ROOT_DIR}/outputs/eval/fewshot/qwen3-vl-4b-ie-4gpu}}" \
  exec bash scripts/evaluate_fewshot.sh "$@"
