"""The single COCO in-context detection protocol used by this repository.

Training-data construction and model evaluation must agree on the prompt.  The
functions in this module are deliberately dependency-light and are imported by
both paths; do not duplicate the prompt in a shell script or trainer subclass.
"""

from __future__ import annotations

import json
from typing import Iterable, Mapping, Sequence

PROTOCOL_NAME = "positive_category_conditioned_icl"
PROMPT_TEMPLATE_VERSION = "inst-v3"
LOSS_MODE = "last_assistant"

SYSTEM_PROMPT = (
    "You are an object detection assistant. "
    "The following image-question-answer pairs are in-context examples. "
    "Each assistant response gives the correct detection result for the "
    "preceding support image. Use these examples to infer how to detect the "
    "requested categories in the final query image. Output only the final "
    "query result in the requested JSON format, using 0-1000 normalized "
    "coordinates. Return detections in descending confidence order with a "
    "score from 0.0 to 1.0; output [] when no requested object is present."
)


def build_question(category: str, *, query: bool = False) -> str:
    """Build the canonical category-conditioned grounding question."""
    if not isinstance(category, str) or not category.strip():
        raise ValueError("category must be a non-empty string")
    category = category.strip()
    prefix = "Using the preceding in-context examples, " if query else ""
    image_phrase = "the query image" if query else "the image"
    verb = "locate" if query else "Locate"
    return (
        f"{prefix}{verb} all of the following objects: {category} in "
        f"{image_phrase} and output at most 20 detections as a confidence-ranked "
        "JSON list like "
        '[{"bbox_2d":[x1,y1,x2,y2],"label":"class_name","score":0.95}]. '
        "The score must be a number from 0.0 to 1.0."
    )


def format_answer(category: str, boxes: Iterable[Sequence[float]]) -> str:
    """Serialize normalized xyxy boxes in the model's expected JSON format."""
    return json.dumps(
        [
            {
                "bbox_2d": [
                    int(value) if float(value).is_integer() else float(value)
                    for value in box
                ],
                "label": category,
                "score": 1.0,
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

    conversations = [{"from": "system", "value": SYSTEM_PROMPT}]
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

    This is the same conversation represented by :func:`build_sft_record`,
    except that the final query answer is omitted and image pixel budgets are
    attached explicitly for the processor.
    """
    category = record.get("category")
    support = record.get("support")
    query = record.get("query")
    if not isinstance(category, str) or not isinstance(support, list) or not isinstance(query, dict):
        raise ValueError("evaluation record needs category, support, and query fields")
    if min_pixels < 1 or max_pixels < min_pixels:
        raise ValueError("invalid evaluation pixel budget")

    messages: list[dict] = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
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
                        {"type": "text", "text": build_question(category, query=True)},
            ],
        }
    )
    return messages


def validate_sft_record(record: Mapping[str, object], *, index: int = 0) -> None:
    """Fail early when a generated or hand-written SFT record is malformed."""
    required = ("image", "conversations")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"record {index} is missing fields: {', '.join(missing)}")
    images = record["image"]
    conversations = record["conversations"]
    if not isinstance(images, (str, list)) or not isinstance(conversations, list):
        raise ValueError(f"record {index} has invalid image/conversations fields")
    if not conversations:
        raise ValueError(f"record {index} has no conversation turns")
    for turn_index, turn in enumerate(conversations):
        if not isinstance(turn, dict) or not isinstance(turn.get("from"), str):
            raise ValueError(f"record {index} turn {turn_index} is malformed")
        if not isinstance(turn.get("value"), str):
            raise ValueError(f"record {index} turn {turn_index} value must be text")
