#!/usr/bin/env python3
"""Compatibility launcher for the few-shot evaluation application.

The implementation lives in ``qwen3vl_sft.evaluation.fewshot``; this file is
kept because existing shell recipes call ``tools/evaluate_fewshot.py``.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen3vl_sft.evaluation.fewshot.cli import (
    GenerationRunner,
    _build_manifest,
    _dataset_dir,
    _dataset_image_roots,
    _embed_detpo_descriptions,
    _evaluate_one,
    _generation_worker_loop,
    _load_model,
    _max_new_tokens_for_dataset,
    _resolve_detpo_prompt_path,
    _result_is_complete,
    main,
    parse_args,
)
from qwen3vl_sft.evaluation.fewshot.config import (
    DATASET_DIRS,
    DATASET_MAX_NEW_TOKENS,
    DEFAULT_DATASETS,
)

__all__ = [
    "DATASET_DIRS",
    "DATASET_MAX_NEW_TOKENS",
    "DEFAULT_DATASETS",
    "GenerationRunner",
    "_build_manifest",
    "_dataset_dir",
    "_dataset_image_roots",
    "_embed_detpo_descriptions",
    "_evaluate_one",
    "_generation_worker_loop",
    "_load_model",
    "_max_new_tokens_for_dataset",
    "_resolve_detpo_prompt_path",
    "_result_is_complete",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    main()
