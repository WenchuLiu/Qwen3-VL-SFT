"""Pure multimodal rendering and label preprocessing.

Dataset indexing and collation live under :mod:`qwen3vl_sft.train` so this
module only transforms one validated record at a time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import torch

from .messages import build_messages
from .rope import get_qwen3_rope_index
from .schema import validate_sft_record


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


