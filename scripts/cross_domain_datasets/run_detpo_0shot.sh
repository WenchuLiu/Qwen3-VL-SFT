#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Defaults to all six registered cross-domain datasets. The common launcher
# retains 640000 max pixels/image and dataset-specific generation budgets.
export DETPO=1
export SHOTS=0
export CATEGORY_DESCRIPTIONS="${CATEGORY_DESCRIPTIONS:-${ROOT_DIR}/docs/cross_domain_category_descriptions.json}"
export WORK_ROOT="${WORK_ROOT:-${OUTPUT_DIR:-${ROOT_DIR}/outputs/eval/fewshot/qwen3-vl-4b-detpo-0shot}}"
exec bash "${ROOT_DIR}/scripts/evaluate_fewshot.sh" "$@"
