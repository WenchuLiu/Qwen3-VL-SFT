#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible path. Use scripts/evaluate_coco_benchmark.sh for new runs.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
exec bash scripts/evaluate_coco_benchmark.sh "$@"
