#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible path. New code should use scripts/evaluate_fewshot.sh
# with VE=1; keep this wrapper so existing VE commands continue to work.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
VE=1 WORK_ROOT="${WORK_ROOT:-${OUTPUT_DIR:-${ROOT_DIR}/outputs/eval/fewshot/qwen3-vl-4b-ve-4gpu}}" \
  exec bash scripts/evaluate_fewshot.sh "$@"
