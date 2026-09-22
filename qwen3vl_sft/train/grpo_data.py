"""Prompt-only datasets for multimodal GRPO training.

The SFT dataset already contains all of the information needed for a rollout:
the conversation up to the final answer and the final answer's ground-truth
boxes.  This module turns those records into a prompt without the final
assistant turn and attaches target metadata for the reward functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

from torch.utils.data import Dataset

from ..data.messages import build_messages
from ..data.paths import resolve_media_path
from ..data.processing import configure_processor_pixels
from ..evaluation.coco.protocol import build_eval_messages, build_question
from .dataset import load_records
from .rewards import parse_target_boxes


def _resolve_media(value: object, base_path: Path) -> object:
    return resolve_media_path(value, base_path)


def _resolve_message_media(messages: Sequence[Mapping[str, object]], base_path: Path) -> list[dict]:
    resolved = []
    for message in messages:
        content = message.get("content", "")
        if not isinstance(content, list):
            resolved.append({**message})
            continue
        parts = []
        for part in content:
            if not isinstance(part, Mapping):
                parts.append(part)
                continue
            item = dict(part)
            if item.get("type") == "image" and "image" in item:
                item["image"] = _resolve_media(item["image"], base_path)
            if item.get("type") == "video" and "video" in item:
                item["video"] = _resolve_media(item["video"], base_path)
            parts.append(item)
        resolved.append({**message, "content": parts})
    return resolved


def _record_boxes(record: Mapping[str, object]) -> list:
    """Read target boxes from an episode, explicit metadata, or its answer."""
    def normalize(value: object) -> list:
        if not isinstance(value, list):
            return []
        if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
            return [value]
        return list(value)

    query = record.get("query")
    if isinstance(query, Mapping) and isinstance(query.get("boxes"), list):
        return normalize(query["boxes"])
    for key in ("target_boxes", "boxes"):
        if isinstance(record.get(key), list):
            return normalize(record[key])

    conversations = record.get("conversations")
    if isinstance(conversations, list):
        for turn in reversed(conversations):
            if not isinstance(turn, Mapping) or turn.get("from") not in {"gpt", "assistant"}:
                continue
            answer = turn.get("value")
            if isinstance(answer, str):
                return [list(box) for box in parse_target_boxes(answer)]
    solution = record.get("solution")
    if solution is not None:
        return [list(box) for box in parse_target_boxes(solution)]
    return []


def _record_labels(
    record: Mapping[str, object], category: str | None, boxes: Sequence[object]
) -> list[str]:
    query = record.get("query")
    if isinstance(query, Mapping) and isinstance(query.get("labels"), list):
        return [str(label) for label in query["labels"]]
    labels = record.get("target_labels")
    if isinstance(labels, list):
        return [str(label) for label in labels]
    return [category or ""] * len(boxes)


def _score_prompt_from_sft_record(
    record: Mapping[str, object],
    base_path: Path,
) -> list[dict]:
    messages = build_messages(record, base_path=base_path)
    last_assistant = -1
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            last_assistant = index
            break
    if last_assistant >= 0:
        messages = messages[:last_assistant]
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("GRPO SFT records must end with a user prompt before the target answer")

    category = record.get("category")
    if isinstance(category, str) and category.strip():
        score_question = build_question(
            category,
            query=True,
            include_confidence=True,
        )
    else:
        score_question = (
            "Return the detections as a JSON list with bbox_2d, label, and a score "
            "field between 0.0 and 1.0, ordered by descending score."
        )

    final_user = dict(messages[-1])
    content = final_user.get("content")
    if isinstance(content, list):
        content = [
            (
                {**part, "text": score_question}
                if isinstance(part, Mapping) and part.get("type") == "text"
                else part
            )
            for part in content
        ]
        final_user["content"] = content
    elif isinstance(content, str):
        final_user["content"] = score_question
    messages[-1] = final_user
    return messages


def _prompt_from_generic_record(record: Mapping[str, object], base_path: Path) -> list[dict]:
    prompt = record.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        problem = record.get("problem", record.get("question"))
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(
                "a GRPO record requires either conversations, support/query, prompt, or problem"
            )
        prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": problem},
                ],
            }
        ]
    media = record.get("image", record.get("images", []))
    if not isinstance(media, list):
        media = [media] if media is not None else []
    media_index = 0
    messages = []
    for index, message in enumerate(prompt):
        if not isinstance(message, Mapping):
            raise ValueError(f"prompt message {index} must be an object")
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported prompt role: {role!r}")
        content = message.get("content", "")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if not isinstance(content, list):
            raise ValueError(f"prompt message {index} content must be text or a list")
        resolved_content = []
        for part in content:
            if not isinstance(part, Mapping):
                raise ValueError(f"prompt message {index} content item must be an object")
            if (
                part.get("type") == "image"
                and "image" not in part
                and media_index < len(media)
            ):
                part = {**part, "image": _resolve_media(media[media_index], base_path)}
                media_index += 1
            elif (
                part.get("type") == "video"
                and "video" not in part
                and media_index < len(media)
            ):
                part = {**part, "video": _resolve_media(media[media_index], base_path)}
                media_index += 1
            resolved_content.append(dict(part))
        messages.append({"role": role, "content": resolved_content})
    while messages and messages[-1].get("role") == "assistant":
        messages.pop()
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("a GRPO prompt must end with a user message")
    final_content = messages[-1].get("content", [])
    if isinstance(final_content, list):
        has_score_instruction = any(
            isinstance(part, Mapping)
            and part.get("type") == "text"
            and any(
                marker in str(part.get("text", "")).lower()
                for marker in ("score", "confidence")
            )
            for part in final_content
        )
        if not has_score_instruction:
            final_content.append(
                {
                    "type": "text",
                    "text": (
                        " Return every detection as JSON with bbox_2d, label, and "
                        "a score from 0.0 to 1.0."
                    ),
                }
            )
    return _resolve_message_media(messages, base_path)


def make_grpo_sample(
    record: Mapping[str, object],
    *,
    base_path: str | Path = ".",
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> dict:
    """Convert one supported annotation record to a rollout sample."""
    base_path = Path(base_path).expanduser().resolve()
    category = record.get("category")
    category = category.strip() if isinstance(category, str) else None

    if isinstance(record.get("support"), list) and isinstance(record.get("query"), Mapping):
        episode = dict(record)
        support = [
            {**frame, "image": _resolve_media(frame.get("image"), base_path)}
            for frame in record["support"]
            if isinstance(frame, Mapping)
        ]
        query = dict(record["query"])
        query["image"] = _resolve_media(query.get("image"), base_path)
        episode["support"] = support
        episode["query"] = query
        if min_pixels is None or max_pixels is None:
            raise ValueError("episode GRPO samples require min_pixels and max_pixels")
        prompt = build_eval_messages(
            episode,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        boxes = _record_boxes(episode)
    elif isinstance(record.get("conversations"), list):
        prompt = _score_prompt_from_sft_record(record, base_path)
        boxes = _record_boxes(record)
    else:
        prompt = _prompt_from_generic_record(record, base_path)
        boxes = _record_boxes(record)

    if category is None:
        category_value = None
    else:
        category_value = category
    return {
        "id": str(record.get("id", record.get("record_id", ""))),
        "prompt": prompt,
        "target_boxes": boxes,
        "target_labels": _record_labels(record, category_value, boxes),
        "category": category_value or "",
        "solution": record.get("solution", ""),
    }


class GRPOPromptDataset(Dataset):
    """Lazy-compatible prompt dataset with reward metadata attached."""

    def __init__(self, processor, paths: Iterable[str | Path], data_args):
        configure_processor_pixels(processor, data_args)
        self.samples: list[dict] = []
        data_root = (
            Path(data_args.data_root).expanduser().resolve()
            if data_args.data_root
            else None
        )
        for raw_path in paths:
            path = Path(raw_path).expanduser().resolve()
            records = load_records(path)
            base_path = data_root or path.parent
            for raw_record in records:
                record = dict(raw_record)
                item_base = record.pop("data_path", None)
                if item_base:
                    item_path = Path(item_base).expanduser()
                    base = (
                        item_path.resolve()
                        if item_path.is_absolute()
                        else (base_path / item_path).resolve()
                    )
                else:
                    base = base_path
                self.samples.append(
                    make_grpo_sample(
                        record,
                        base_path=base,
                        min_pixels=data_args.min_pixels,
                        max_pixels=data_args.max_pixels,
                    )
                )
        if not self.samples:
            raise ValueError("no GRPO training records were loaded")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


def grpo_data_collator(features: Sequence[Mapping[str, object]]) -> list[dict]:
    """GRPO performs multimodal rendering inside the trainer."""
    return [dict(feature) for feature in features]


__all__ = [
    "GRPOPromptDataset",
    "grpo_data_collator",
    "make_grpo_sample",
]
