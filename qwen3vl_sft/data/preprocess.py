"""Compatibility facade for the pre-refactor data module.

The implementation is now split between ``data/processing.py`` (single-record
rendering) and ``train/{dataset,collator,data}.py`` (dataset orchestration).
"""

from .processing import (
    IGNORE_INDEX,
    _encode,
    _find_subsequence,
    _grid_values,
    _token_id,
    configure_processor_pixels,
    find_assistant_spans,
    preprocess_record,
)
from ..train.collator import MultimodalDataCollator
from ..train.data import make_data_module
from ..train.dataset import SupervisedDataset, _load_records, _split_dataset

__all__ = [
    "IGNORE_INDEX",
    "MultimodalDataCollator",
    "SupervisedDataset",
    "_load_records",
    "_split_dataset",
    "_encode",
    "_find_subsequence",
    "_grid_values",
    "_token_id",
    "configure_processor_pixels",
    "find_assistant_spans",
    "make_data_module",
    "preprocess_record",
]
