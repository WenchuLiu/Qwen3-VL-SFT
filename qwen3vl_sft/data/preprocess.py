"""Qwen3-VL chat-template preprocessing and multimodal batch collation."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from torch.utils.data import Dataset, Subset

from .messages import build_messages
from .rope import get_qwen3_rope_index
from ..evaluation.coco_protocol import validate_sft_record


IGNORE_INDEX = -100


def _token_id(tokenizer, token: str, fallback: int | None = None) -> int | None:
    try:
        value = tokenizer.convert_tokens_to_ids(token)
    except (KeyError, TypeError, ValueError):
        return fallback
    unknown = getattr(tokenizer, "unk_token_id", None)
    if value is None or value == unknown:
        return fallback
    return int(value)


def _encode(tokenizer, text: str) -> list[int]:
    try:
        values = tokenizer.encode(text, add_special_tokens=False)
    except (AttributeError, TypeError, ValueError):
        return []
    return [int(value) for value in values]


def _find_subsequence(values: Sequence[int], pattern: Sequence[int], start: int = 0) -> int:
    if not pattern:
        return -1
    last = len(values) - len(pattern) + 1
    for index in range(start, max(start, last)):
        if list(values[index:index + len(pattern)]) == list(pattern):
            return index
    return -1


def find_assistant_spans(input_ids: torch.Tensor, tokenizer) -> list[tuple[int, int]]:
    """Find assistant response spans from chat-template role markers.

    Hard-coding Qwen token IDs is fragile across tokenizer revisions.  We first
    search encoded ``<|im_start|>assistant\n`` and ``<|im_end|>`` sequences,
    then use a narrow role-marker fallback for older templates.
    """
    row = input_ids[0].tolist()
    prefix = _encode(tokenizer, "<|im_start|>assistant\n")
    end_marker = _encode(tokenizer, "<|im_end|>")
    if not end_marker:
        eos = getattr(tokenizer, "eos_token_id", None)
        if isinstance(eos, int):
            end_marker = [eos]

    starts = []
    if prefix:
        cursor = 0
        while True:
            position = _find_subsequence(row, prefix, cursor)
            if position < 0:
                break
            starts.append(position + len(prefix))
            cursor = position + len(prefix)

    if not starts:
        im_start = _encode(tokenizer, "<|im_start|>")
        assistant = _encode(tokenizer, "assistant")
        newline = _encode(tokenizer, "\n")
        role_marker = [*im_start, *assistant, *newline]
        cursor = 0
        while role_marker:
            position = _find_subsequence(row, role_marker, cursor)
            if position < 0:
                break
            starts.append(position + len(role_marker))
            cursor = position + len(role_marker)

    spans = []
    for answer_start in starts:
        answer_end = _find_subsequence(row, end_marker, answer_start)
        if answer_end < 0:
            answer_end = len(row)
        else:
            answer_end += len(end_marker)
        if answer_end > answer_start:
            spans.append((answer_start, answer_end))
    return spans


def configure_processor_pixels(processor, data_args):
    """Apply one image/video budget to the processor used by training."""
    image_processor = processor.image_processor
    if hasattr(image_processor, "min_pixels"):
        image_processor.min_pixels = data_args.min_pixels
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = data_args.max_pixels
    if isinstance(getattr(image_processor, "size", None), dict):
        image_processor.size["shortest_edge"] = data_args.min_pixels
        image_processor.size["longest_edge"] = data_args.max_pixels

    video_processor = getattr(processor, "video_processor", None)
    if video_processor is not None:
        video_values = {
            "min_pixels": "video_min_pixels",
            "max_pixels": "video_max_pixels",
            "min_frames": "video_min_frames",
            "max_frames": "video_max_frames",
            "fps": "video_fps",
        }
        for name, value_name in video_values.items():
            if hasattr(video_processor, name) and hasattr(data_args, value_name):
                setattr(video_processor, name, getattr(data_args, value_name))
        if isinstance(getattr(video_processor, "size", None), dict):
            video_processor.size["shortest_edge"] = data_args.video_min_pixels
            video_processor.size["longest_edge"] = data_args.video_max_pixels
    return processor


def _grid_values(value) -> list[torch.Tensor] | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return [value.view(1, -1)] if value.ndim == 1 else [value]
    if isinstance(value, (list, tuple)):
        tensors = []
        for item in value:
            tensor = item if isinstance(item, torch.Tensor) else torch.as_tensor(item)
            tensors.append(tensor.view(1, -1) if tensor.ndim == 1 else tensor)
        return tensors
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return [tensor.view(1, -1) if tensor.ndim == 1 else tensor]


def preprocess_record(
    record: Mapping[str, object],
    processor,
    *,
    base_path: str | Path = ".",
) -> dict[str, torch.Tensor | object]:
    """Render one record, create assistant-only labels, and compute RoPE IDs."""
    validate_sft_record(record)
    messages = build_messages(record, base_path=base_path)
    rendered = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    values = dict(rendered)
    input_ids = values["input_ids"]
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    values["input_ids"] = input_ids

    labels = torch.full_like(input_ids, IGNORE_INDEX)
    spans = find_assistant_spans(input_ids, processor.tokenizer)
    if not spans:
        raise ValueError(
            "chat template produced no assistant span; check the tokenizer "
            "template or use a supported Qwen3-VL Transformers version"
        )
    loss_mode = record.get("loss_mode", "all_assistant")
    if loss_mode == "last_assistant":
        spans = spans[-1:]
    elif loss_mode != "all_assistant":
        raise ValueError("loss_mode must be 'all_assistant' or 'last_assistant'")
    for start, end in spans:
        labels[0, start:end] = input_ids[0, start:end]
    if int((labels != IGNORE_INDEX).sum().item()) == 0:
        raise ValueError("record produced zero supervised tokens")
    values["labels"] = labels

    image_grid = _grid_values(values.get("image_grid_thw"))
    video_grid = _grid_values(values.get("video_grid_thw"))
    tokenizer = processor.tokenizer
    image_token_id = _token_id(tokenizer, "<|image_pad|>", 151655)
    video_token_id = _token_id(tokenizer, "<|video_pad|>", 151656)
    vision_start_token_id = _token_id(tokenizer, "<|vision_start|>", 151652)
    position_ids, _ = get_qwen3_rope_index(
        spatial_merge_size=int(getattr(processor.image_processor, "merge_size", 2)),
        input_ids=input_ids,
        image_grid_thw=torch.cat(image_grid, dim=0) if image_grid else None,
        video_grid_thw=torch.cat(video_grid, dim=0) if video_grid else None,
        image_token_id=151655 if image_token_id is None else image_token_id,
        video_token_id=151656 if video_token_id is None else video_token_id,
        vision_start_token_id=151652 if vision_start_token_id is None else vision_start_token_id,
    )
    values["position_ids"] = position_ids
    return values


def _load_records(path: Path) -> list[dict]:
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
            records = _load_records(path)
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


def _split_dataset(dataset: Dataset, ratio: float, seed: int) -> tuple[Dataset, Dataset]:
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


@dataclass
class MultimodalDataCollator:
    tokenizer: object
    model_max_length: int

    def __call__(self, instances: Sequence[Mapping[str, object]]) -> dict[str, object]:
        if not instances:
            raise ValueError("cannot collate an empty batch")
        input_ids = []
        labels = []
        positions = []
        for instance in instances:
            ids = instance["input_ids"]
            target = instance["labels"]
            position = instance["position_ids"]
            if ids.shape[-1] > self.model_max_length:
                raise ValueError(
                    "a sample exceeds model_max_length; increase --model-max-length "
                    "because truncating multimodal token sequences can desynchronize "
                    "image/video tokens and their pixel grids"
                )
            input_ids.append(ids.squeeze(0))
            labels.append(target.squeeze(0))
            positions.append(position)

        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, "eos_token_id", 0)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        max_length = input_ids.shape[1]
        padded_positions = [
            torch.nn.functional.pad(
                position[..., :max_length],
                (0, max_length - position.shape[-1]),
                value=1,
            )
            for position in positions
        ]
        batch: dict[str, object] = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(pad_token_id),
            "position_ids": torch.cat(padded_positions, dim=1),
        }

        for input_key, output_key in (
            ("pixel_values", "pixel_values"),
            ("image_grid_thw", "image_grid_thw"),
            ("pixel_values_videos", "pixel_values_videos"),
            ("video_grid_thw", "video_grid_thw"),
        ):
            values = [instance.get(input_key) for instance in instances]
            values = [value for value in values if value is not None]
            if values:
                batch[output_key] = torch.cat(values, dim=0)
        return batch


def make_data_module(processor, data_args) -> dict:
    train_dataset = SupervisedDataset(processor, data_args.dataset, data_args)
    eval_dataset = None
    if data_args.eval_mode == "loss":
        if data_args.eval_dataset:
            eval_dataset = SupervisedDataset(processor, data_args.eval_dataset, data_args)
        else:
            train_dataset, eval_dataset = _split_dataset(
                train_dataset, data_args.eval_ratio, data_args.eval_seed
            )
    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": MultimodalDataCollator(
            tokenizer=processor.tokenizer,
            model_max_length=data_args.model_max_length,
        ),
    }
