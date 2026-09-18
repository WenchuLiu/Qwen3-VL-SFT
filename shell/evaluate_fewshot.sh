#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "${ROOT_DIR}"
exec "${PYTHON_BIN}" tools/evaluate_fewshot.py "$@"
