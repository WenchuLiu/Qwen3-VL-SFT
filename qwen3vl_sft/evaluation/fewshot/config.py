"""Dataset registry and result-file validation for few-shot evaluation."""

from __future__ import annotations

import json
from pathlib import Path


DATASET_DIRS = {
    "ArTaxOr": "ArTaxOr",
    "Clipart1k": "clipart1k",
    "FISH": "FISH",
    "NEU-DET": "NEU-DET",
    "UODD": "UODD",
    "DIOR": "VISUALDIOR",
    "VISUALDIOR": "VISUALDIOR",
}

DATASET_MAX_NEW_TOKENS = {
    "ArTaxOr": 1024,
    "Clipart1k": 1024,
    "FISH": 1024,
    "NEU-DET": 1024,
    "UODD": 1024,
    "DIOR": 2048,
    "VISUALDIOR": 2048,
}

DEFAULT_DATASETS = (
    "ArTaxOr",
    "Clipart1k",
    "FISH",
    "NEU-DET",
    "UODD",
    "VISUALDIOR",
)


def max_new_tokens_for_dataset(dataset_name: str, override: int | None) -> int:
    """Return the dataset default unless an explicit override is supplied."""
    if override is not None:
        return override
    try:
        return DATASET_MAX_NEW_TOKENS[dataset_name]
    except KeyError as error:
        raise ValueError(f"no max_new_tokens configured for {dataset_name!r}") from error


def dataset_dir(data_root: Path, name: str) -> tuple[str, Path]:
    """Resolve a case-insensitive dataset name and verify its directory."""
    if name in DATASET_DIRS:
        display_name = name
        directory = data_root / DATASET_DIRS[name]
    else:
        matches = {key.lower(): key for key in DATASET_DIRS}
        canonical = matches.get(name.lower())
        if canonical is None:
            valid = ", ".join(DATASET_DIRS)
            raise ValueError(f"unknown dataset {name!r}; choose from: {valid}")
        display_name = canonical
        directory = data_root / DATASET_DIRS[canonical]
    if not directory.is_dir():
        raise NotADirectoryError(f"dataset directory not found: {directory}")
    return display_name, directory


def dataset_image_roots(dataset_path: Path) -> tuple[Path, Path]:
    """Resolve support/query image roots in either supported local layout."""
    support_root = dataset_path / "new_train"
    query_root = dataset_path / "new_test"
    if not support_root.is_dir():
        support_root = dataset_path / "train"
    if not query_root.is_dir():
        query_root = dataset_path / "test"
    return support_root, query_root


def result_is_complete(path: Path, expected_max_new_tokens: int | None = None) -> bool:
    """Return whether a cached result has the fields needed for reuse."""
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not bool(payload.get("metrics_by_shot")) or "predictions" not in payload:
        return False
    return (
        expected_max_new_tokens is None
        or payload.get("max_new_tokens") == expected_max_new_tokens
    )


# Compatibility spellings used by the original CLI.
_max_new_tokens_for_dataset = max_new_tokens_for_dataset
_dataset_dir = dataset_dir
_dataset_image_roots = dataset_image_roots
_result_is_complete = result_is_complete
