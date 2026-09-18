"""Dataset indexing and deterministic train/eval splitting."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset, Subset

from ..data.processing import configure_processor_pixels, preprocess_record


def load_records(path: Path) -> list[dict]:
    """Load either a JSON list/object or a JSONL annotation file."""
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in handle if line.strip()]
        else:
            payload = json.load(handle)
            records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        raise ValueError(f"dataset {path} contains no records")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"dataset {path} must contain JSON objects")
    return records


class SupervisedDataset(Dataset):
    """Lazy multimodal dataset for the official Qwen conversation schema."""

    def __init__(self, processor, paths: Iterable[str | Path], data_args):
        self.processor = configure_processor_pixels(processor, data_args)
        self.data_args = data_args
        self.samples: list[tuple[dict, Path]] = []
        data_root = (
            Path(data_args.data_root).expanduser().resolve()
            if data_args.data_root
            else None
        )
        for raw_path in paths:
            path = Path(raw_path).expanduser().resolve()
            records = load_records(path)
            base_path = data_root or path.parent
            for record in records:
                item = dict(record)
                item_base = item.pop("data_path", None)
                if item_base:
                    item_path = Path(item_base).expanduser()
                    sample_base = (
                        item_path.resolve()
                        if item_path.is_absolute()
                        else (base_path / item_path).resolve()
                    )
                else:
                    sample_base = base_path
                self.samples.append((item, sample_base))
        if not self.samples:
            raise ValueError("no training records were loaded")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | object]:
        record, base_path = self.samples[index]
        return preprocess_record(record, self.processor, base_path=base_path)


def split_dataset(dataset: Dataset, ratio: float, seed: int) -> tuple[Dataset, Dataset | None]:
    """Split a dataset deterministically without allowing an empty train set."""
    if not 0 <= ratio < 1:
        raise ValueError("eval_ratio must be in [0, 1)")
    if ratio == 0:
        return dataset, None
    if len(dataset) < 2:
        raise ValueError("at least two samples are required for an eval split")
    eval_size = min(len(dataset) - 1, max(1, round(len(dataset) * ratio)))
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    return Subset(dataset, indices[eval_size:]), Subset(dataset, indices[:eval_size])


# Private compatibility name used by the original flat module.
_load_records = load_records
_split_dataset = split_dataset
