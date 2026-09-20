"""Canonical COCO in-context detection prompts used by training and evaluation.

Training teaches box and label generation without synthetic confidence targets.
Evaluation keeps the same support format but asks the final query for confidence
scores.  Both paths live here so their intentional difference remains explicit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from PIL import Image, ImageDraw

from ...data.schema import validate_sft_record

PROTOCOL_NAME = "positive_category_conditioned_icl"
PROMPT_TEMPLATE_VERSION = "inst-v5"
LOSS_MODE = "last_assistant"
VE_BOX_COLOR = (255, 0, 0)

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

ZERO_SHOT_EVAL_SYSTEM_PROMPT = (
    "You are an object detection assistant. "
    "Detect the requested category in the image. "
    "Estimate a confidence score from 0.0 to 1.0 for every detection and "
    "return detections in descending confidence order. Output only the "
    "requested JSON list, using 0-1000 normalized coordinates; output [] "
    "when no requested object is present."
)


def build_question(
    category: str,
    *,
    query: bool = False,
    include_confidence: bool = False,
    category_description: str | None = None,
) -> str:
    """Build the canonical category-conditioned grounding question.

    ``category_description`` is intentionally optional. Keeping it out of the
    default path makes the original baseline prompt byte-for-byte compatible,
    while callers can add a natural-language description for a training-free
    instruction-enhancement experiment.
    """
    if not isinstance(category, str) or not category.strip():
        raise ValueError("category must be a non-empty string")
    category = category.strip()
    if category_description is not None:
        if not isinstance(category_description, str) or not category_description.strip():
            raise ValueError("category_description must be a non-empty string when provided")
        category_description = category_description.strip()
    prefix = "Using the preceding in-context examples, " if query else ""
    image_phrase = "the query image" if query else "the image"
    verb = "locate" if query else "Locate"
    if category_description is not None:
        question = (
            f"{prefix}{verb} all of the following objects: {category} in "
            f"{image_phrase}.\n"
            f"{category} is {category_description}.\n"
            "Output all detections as a JSON list like "
        )
    else:
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


def load_category_descriptions(path: str | Path) -> dict[str, str]:
    """Load a category-to-description mapping from a JSON file.

    Both of these forms are accepted so a small ad-hoc mapping and a versioned
    experiment config are equally convenient::

        {"fish": "an aquatic animal ..."}
        {"descriptions": {"fish": "an aquatic animal ..."}}
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"category description file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"category description file is not valid JSON: {path}") from error

    if isinstance(payload, dict) and "descriptions" in payload:
        payload = payload["descriptions"]
    if not isinstance(payload, dict) or not payload:
        raise ValueError(
            "category description JSON must be a non-empty object, optionally under "
            "the 'descriptions' key"
        )

    descriptions: dict[str, str] = {}
    for category, description in payload.items():
        if not isinstance(category, str) or not category.strip():
            raise ValueError("category description keys must be non-empty strings")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(
                f"description for category {category!r} must be a non-empty string"
            )
        descriptions[category.strip()] = description.strip()
    return descriptions


def _resolve_category_description(
    category: str,
    *,
    embedded_description: object = None,
    category_descriptions: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve an external description, falling back to an episode field."""
    if category_descriptions:
        description = category_descriptions.get(category)
        if description is None:
            folded_category = category.casefold()
            matches = [
                value
                for key, value in category_descriptions.items()
                if isinstance(key, str) and key.casefold() == folded_category
            ]
            description = matches[0] if matches else None
        if description is not None:
            if not isinstance(description, str) or not description.strip():
                raise ValueError(
                    f"description for category {category!r} must be a non-empty string"
                )
            return description.strip()

    if embedded_description is not None:
        if not isinstance(embedded_description, str) or not embedded_description.strip():
            raise ValueError(
                f"embedded description for category {category!r} must be a non-empty string"
            )
        return embedded_description.strip()
    return None


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


def _draw_visual_enhancement_boxes(
    frame: Mapping[str, object],
    *,
    line_width: int = 4,
) -> Image.Image:
    """Load a support image and draw its normalized GT boxes in red.

    VE is deliberately applied only to support frames.  The episode stores
    boxes in Qwen's 0-1000 coordinate system, while PIL expects pixels, so
    the frame's original width/height are used for the conversion.  Returning
    a PIL image is supported by ``qwen-vl-utils`` and avoids writing temporary
    annotated files into the dataset or evaluation output directories.
    """
    image_value = _frame_image(frame)
    image_value = image_value.removeprefix("file://")
    if "://" in image_value or image_value.startswith("data:"):
        raise ValueError(
            "visual enhancement requires local support images; "
            f"cannot annotate {image_value!r}"
        )

    try:
        with Image.open(image_value) as source:
            annotated = source.convert("RGB").copy()
    except (OSError, ValueError) as error:
        raise ValueError(
            f"could not load local support image for visual enhancement: {image_value}"
        ) from error

    try:
        frame_width = float(frame.get("width", annotated.width))
        frame_height = float(frame.get("height", annotated.height))
    except (TypeError, ValueError):
        frame_width = float(annotated.width)
        frame_height = float(annotated.height)
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(
            "support frame width and height must be positive for visual enhancement"
        )

    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size
    line_width = max(1, int(line_width))
    for box in _frame_boxes(frame):
        try:
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = [float(value) for value in box]
        except (TypeError, ValueError):
            continue
        x1 = max(0, min(width - 1, round(x1 / 1000.0 * frame_width)))
        y1 = max(0, min(height - 1, round(y1 / 1000.0 * frame_height)))
        x2 = max(0, min(width - 1, round(x2 / 1000.0 * frame_width)))
        y2 = max(0, min(height - 1, round(y2 / 1000.0 * frame_height)))
        if x2 <= x1 or y2 <= y1:
            continue
        draw.rectangle((x1, y1, x2, y2), outline=VE_BOX_COLOR, width=line_width)
    return annotated


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
    visual_enhancement: bool = False,
    instruction_enhancement: bool = False,
    category_descriptions: Mapping[str, str] | None = None,
) -> list[dict]:
    """Build the generation prompt for one fixed episode.

    Support turns retain the score-free SFT schema.  The final query answer is
    omitted and its prompt explicitly requests model-estimated confidence.
    When ``visual_enhancement`` is enabled, red ground-truth boxes are drawn
    on support images only; the query image is never annotated. When
    ``instruction_enhancement`` is enabled, the target category's description
    is added to the final query question. With zero shots, this is the only
    user turn; with support shots, support questions remain in the original
    score-free SFT format. Descriptions may be passed in
    ``category_descriptions`` or embedded in the episode as
    ``category_description``.
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
    category_description = None
    if instruction_enhancement:
        category_description = _resolve_category_description(
            category,
            embedded_description=record.get("category_description"),
            category_descriptions=category_descriptions,
        )
        if category_description is None:
            raise ValueError(
                f"instruction enhancement requires a description for category {category!r}; "
                "provide --category-descriptions or add category_description to the episode"
            )

    zero_shot = not support
    messages: list[dict] = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        ZERO_SHOT_EVAL_SYSTEM_PROMPT
                        if zero_shot
                        else EVAL_SYSTEM_PROMPT
                    ),
                }
            ],
        }
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
                            "image": (
                                _draw_visual_enhancement_boxes(frame)
                                if visual_enhancement
                                else _frame_image(frame)
                            ),
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
                        query=not zero_shot,
                        include_confidence=True,
                        category_description=category_description,
                    ),
                },
            ],
        }
    )
    return messages
