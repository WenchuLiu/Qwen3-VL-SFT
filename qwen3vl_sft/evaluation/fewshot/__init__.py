"""Cross-domain few-shot detection evaluation."""

from .config import (
    DATASET_DIRS,
    DATASET_MAX_NEW_TOKENS,
    DEFAULT_DATASETS,
    dataset_dir,
    max_new_tokens_for_dataset,
    result_is_complete,
)

__all__ = [
    "DATASET_DIRS",
    "DATASET_MAX_NEW_TOKENS",
    "DEFAULT_DATASETS",
    "dataset_dir",
    "max_new_tokens_for_dataset",
    "result_is_complete",
]
