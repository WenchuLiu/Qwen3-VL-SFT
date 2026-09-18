"""Canonical COCO in-context detection prompts used by training and evaluation.

Training teaches box and label generation without synthetic confidence targets.
Evaluation keeps the same support format but asks the final query for confidence
scores.  Both paths live here so their intentional difference remains explicit.
"""

from __future__ import annotations

import json
from typing import Iterable, Mapping, Sequence

from ...data.schema import validate_sft_record

PROTOCOL_NAME = "positive_category_conditioned_icl"
PROMPT_TEMPLATE_VERSION = "inst-v5"
LOSS_MODE = "last_assistant"

TRAIN_SYSTEM_PROMPT = (
    "You are an object detection assistant. "
    "The following image-question-answer pairs are in-context examples. "
    "Each assistant response gives the correct detection result for the "
    "preceding support image. Use these examples to infer how to detect the "
    "requested categories in the final query image. Output only the final "
    "query result in the requested JSON format, using 0-1000 normalized "
    "coordinates. Output [] when no requested object is present."
)

EVAL_SYSTEM_PROMPT = (
    "You are an object detection assistant. "
    "The following image-question-answer pairs are in-context examples. "
    "Their answers demonstrate the box and label format without confidence "
    "scores. Use these examples to detect the requested category in the final "
    "query image. For the final query only, estimate a confidence score from "
    "0.0 to 1.0 for every detection and return detections in descending "
    "confidence order. Output only the requested JSON list, using 0-1000 "
    "normalized coordinates; output [] when no requested object is present."
)


def build_question(
    category: str,
    *,
    query: bool = False,
    include_confidence: bool = False,
) -> str:
    """Build the canonical category-conditioned grounding question."""
    if not isinstance(category, str) or not category.strip():
        raise ValueError("category must be a non-empty string")
    category = category.strip()
    prefix = "Using the preceding in-context examples, " if query else ""
    image_phrase = "the query image" if query else "the image"
    verb = "locate" if query else "Locate"
    question = (
        f"{prefix}{verb} all of the following objects: {category} in "
        f"{image_phrase} and output all detections as a JSON list like "
    )
    if include_confidence:
        return (
            question
            + '[{"bbox_2d":[x1,y1,x2,y2],"label":"class_name","score":0.95}]. '
            "Estimate each score from 0.0 to 1.0 and order detections by "
            "descending confidence."
        )
    return question + '[{"bbox_2d":[x1,y1,x2,y2],"label":"class_name"}].'


def format_answer(category: str, boxes: Iterable[Sequence[float]]) -> str:
    """Serialize GT boxes without inventing confidence supervision."""
    return json.dumps(
        [
            {
                "bbox_2d": [
                    int(value) if float(value).is_integer() else float(value)
                    for value in box
                ],
                "label": category,
            }
            for box in boxes
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _frame_image(frame: Mapping[str, object]) -> str:
    image = frame.get("image")
    if not isinstance(image, str) or not image:
        raise ValueError("episode frame requires a non-empty 'image' path")
    return image


def _frame_boxes(frame: Mapping[str, object]) -> Sequence[Sequence[float]]:
    boxes = frame.get("boxes", [])
    if not isinstance(boxes, list):
        raise ValueError("episode frame 'boxes' must be a list")
    return boxes


def build_sft_record(
    *,
    record_id: str,
    category: str,
    support: Sequence[Mapping[str, object]],
    query: Mapping[str, object],
) -> dict:
    """Convert one support/query episode into the official SFT JSON schema.

    The query answer is present in the training record, but ``loss_mode`` makes
    the data collator supervise only this final assistant turn.  Support
    answers remain context and are never treated as targets.
    """
    frames = [*support, query]
    if not frames:
        raise ValueError("an SFT episode must contain a query frame")

    conversations = [{"from": "system", "value": TRAIN_SYSTEM_PROMPT}]
    for index, frame in enumerate(frames):
        is_query = index == len(frames) - 1
        conversations.extend(
            [
                {
                    "from": "human",
                    "value": f"<image>\n{build_question(category, query=is_query)}",
                },
                {"from": "gpt", "value": format_answer(category, _frame_boxes(frame))},
            ]
        )

    return {
        "id": record_id,
        "task": "incontext_object_detection",
        "protocol": PROTOCOL_NAME,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "category": category,
        "num_shots": len(support),
        "image": [_frame_image(frame) for frame in frames],
        "conversations": conversations,
        "loss_mode": LOSS_MODE,
    }


def build_eval_messages(
    record: Mapping[str, object],
    *,
    min_pixels: int,
    max_pixels: int,
) -> list[dict]:
    """Build the generation prompt for one fixed episode.

    Support turns retain the score-free SFT schema.  The final query answer is
    omitted and its prompt explicitly requests model-estimated confidence.
    """
    category = record.get("category")
    support = record.get("support")
    query = record.get("query")
    if (
        not isinstance(category, str)
        or not isinstance(support, list)
        or not isinstance(query, dict)
    ):
        raise ValueError("evaluation record needs category, support, and query fields")
    if min_pixels < 1 or max_pixels < min_pixels:
        raise ValueError("invalid evaluation pixel budget")

    messages: list[dict] = [
        {"role": "system", "content": [{"type": "text", "text": EVAL_SYSTEM_PROMPT}]}
    ]
    support_question = build_question(category)
    for frame in support:
        if not isinstance(frame, dict):
            raise ValueError("support frames must be objects")
        messages.extend(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": _frame_image(frame),
                            "min_pixels": min_pixels,
                            "max_pixels": max_pixels,
                        },
                        {"type": "text", "text": support_question},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": format_answer(category, _frame_boxes(frame))}
                    ],
                },
            ]
        )

    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": _frame_image(query),
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                },
                {
                    "type": "text",
                    "text": build_question(
                        category,
                        query=True,
                        include_confidence=True,
                    ),
                },
            ],
        }
    )
    return messages
