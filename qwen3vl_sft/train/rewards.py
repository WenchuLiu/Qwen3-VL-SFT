"""Reward functions for score-aware visual grounding GRPO training.

The reference Visual-RFT implementation uses three signals for grounding:
the quality of the predicted boxes, the confidence assigned to those boxes,
and the output format.  This module keeps those signals independent so that
they can be enabled, weighted, and unit-tested without importing Trainer or
Transformers.
"""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from ..evaluation.coco.metrics import compute_iou, parse_scored_detection_output


_ANSWER_PATTERN = re.compile(r"<answer>(?P<answer>.*?)</answer>", re.DOTALL)
_SCORE_PATTERN = re.compile(
    r"(?:\"score\"|\'score\'|\"Confidence\"|\'Confidence\')\s*:\s*"
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
)


@dataclass(frozen=True)
class ScoredDetection:
    """A parsed prediction used by the reward functions."""

    label: str
    box: tuple[float, float, float, float]
    score: float
    has_explicit_score: bool = True


def completion_text(completion: object) -> str:
    """Extract text from TRL-style or plain-string completions."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Mapping):
        content = completion.get("content")
        return content if isinstance(content, str) else str(content or "")
    if isinstance(completion, Sequence) and not isinstance(completion, (bytes, bytearray)):
        if not completion:
            return ""
        return completion_text(completion[0])
    return str(completion or "")


def _answer_content(text: str) -> str:
    match = _ANSWER_PATTERN.search(text)
    return match.group("answer").strip() if match else text.strip()


def _valid_box(values: object) -> tuple[float, float, float, float] | None:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != 4:
        return None
    try:
        box = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) and 0.0 <= value <= 1000.0 for value in box):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _score(value: object) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    # Visual-RFT uses confidence in [0, 1].  Do not silently rescale values:
    # a model producing 95 instead of 0.95 should receive a format penalty.
    if not 0.0 <= value <= 1.0:
        return None
    return value


def parse_reward_detections(text: str) -> list[ScoredDetection]:
    """Parse current Qwen JSON and the legacy Visual-RFT bbox format.

    ``parse_scored_detection_output`` intentionally supplies a neutral score
    when a score field is absent.  For RL we additionally retain whether the
    model explicitly emitted the field, because the score reward must not
    reward score-free answers.
    """
    text = _answer_content(text)
    parsed: list[ScoredDetection] = []

    # Visual-RFT examples often use Python-style single-quoted dictionaries.
    # Accept that safe literal subset before trying strict JSON scanning.
    try:
        literal_candidate = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        literal_candidate = None
    if isinstance(literal_candidate, (list, dict)):
        entries = literal_candidate if isinstance(literal_candidate, list) else [literal_candidate]
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            box = _valid_box(entry.get("bbox_2d", entry.get("Position")))
            if box is None:
                continue
            label = entry.get("label", entry.get("Label", ""))
            label = label.strip() if isinstance(label, str) else ""
            raw_score = entry.get("score", entry.get("Confidence"))
            score = _score(raw_score)
            legacy_bbox = "Position" in entry or "Confidence" in entry
            if label or legacy_bbox:
                parsed.append(
                    ScoredDetection(
                        label,
                        box,
                        0.0 if score is None else score,
                        score is not None and raw_score is not None,
                    )
                )
        if isinstance(literal_candidate, list):
            return parsed
        if parsed:
            return parsed

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        entries = candidate if isinstance(candidate, list) else [candidate]
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            box = _valid_box(entry.get("bbox_2d", entry.get("Position")))
            if box is None:
                continue
            label = entry.get("label", entry.get("Label", ""))
            label = label.strip() if isinstance(label, str) else ""
            raw_score = entry.get("score", entry.get("Confidence"))
            score = _score(raw_score)
            legacy_bbox = "Position" in entry or "Confidence" in entry
            if label or legacy_bbox:
                parsed.append(
                    ScoredDetection(
                        label,
                        box,
                        0.0 if score is None else score,
                        score is not None and raw_score is not None,
                    )
                )
        if isinstance(candidate, list):
            return parsed
        if parsed:
            return parsed

    # Keep compatibility with the legacy parser for formats it accepts but
    # whose JSON is embedded in surrounding text.  Such detections have no
    # score reward unless the text visibly contains a confidence field.
    legacy = parse_scored_detection_output(text)
    if legacy:
        explicit = bool(_SCORE_PATTERN.search(text))
        return [
            ScoredDetection(label, tuple(box), score, explicit)
            for label, box, score in legacy
        ]
    return []


def _parse_boxes_from_text(text: str) -> list[tuple[float, float, float, float]]:
    """Parse target boxes even when a legacy solution has no label/score."""
    try:
        literal_candidate = ast.literal_eval(_answer_content(text))
    except (SyntaxError, ValueError):
        literal_candidate = None
    if isinstance(literal_candidate, (list, dict)):
        entries = literal_candidate if isinstance(literal_candidate, list) else [literal_candidate]
        boxes = [
            box
            for entry in entries
            if isinstance(entry, Mapping)
            for box in [_valid_box(entry.get("bbox_2d", entry.get("Position")))]
            if box is not None
        ]
        if isinstance(literal_candidate, list) or boxes:
            return boxes

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        entries = candidate if isinstance(candidate, list) else [candidate]
        boxes = []
        for entry in entries:
            if isinstance(entry, Mapping):
                box = _valid_box(entry.get("bbox_2d", entry.get("Position")))
                if box is not None:
                    boxes.append(box)
        if isinstance(candidate, list) or boxes:
            return boxes
    return []


def _as_boxes(value: object) -> list[tuple[float, float, float, float]]:
    """Normalize common target-box representations."""
    if value is None:
        return []
    if isinstance(value, str):
        return _parse_boxes_from_text(value) or [
            box for _, box, _ in parse_scored_detection_output(value)
        ]
    if isinstance(value, Mapping):
        value = value.get("boxes", value.get("target_boxes", []))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
        box = _valid_box(value)
        return [box] if box is not None else []
    boxes: list[tuple[float, float, float, float]] = []
    for item in value:
        if isinstance(item, Mapping):
            item = item.get("bbox_2d", item.get("Position", item.get("box")))
        box = _valid_box(item)
        if box is not None:
            boxes.append(box)
    return boxes


def parse_target_boxes(value: object) -> list[tuple[float, float, float, float]]:
    """Public target-box normalizer for dataset adapters."""
    return _as_boxes(value)


def _target_for_index(target_boxes: object, index: int) -> list[tuple[float, float, float, float]]:
    """Get one target list from a batched reward argument."""
    if target_boxes is None:
        return []
    if isinstance(target_boxes, Mapping):
        return _as_boxes(target_boxes)
    if not isinstance(target_boxes, Sequence) or isinstance(target_boxes, (str, bytes)):
        return _as_boxes(target_boxes)
    # A single box list is also a valid target.  Distinguish it from a batch
    # by inspecting the first element.
    if not target_boxes:
        return []
    first = target_boxes[0]
    if isinstance(first, (int, float)):
        return _as_boxes(target_boxes)
    if isinstance(first, Mapping):
        return _as_boxes(target_boxes)
    if isinstance(first, Sequence) and len(first) == 4 and all(
        isinstance(item, (int, float)) for item in first
    ):
        return _as_boxes(target_boxes)
    if index >= len(target_boxes):
        return []
    return _as_boxes(target_boxes[index])


def _category_for_index(category: object, index: int) -> str | None:
    if isinstance(category, str):
        return category.strip().casefold() or None
    if isinstance(category, Sequence) and not isinstance(category, (bytes, bytearray, str)):
        if index < len(category) and isinstance(category[index], str):
            return category[index].strip().casefold() or None
    return None


def _labels_for_index(labels: object, index: int) -> list[str]:
    if labels is None:
        return []
    if isinstance(labels, str):
        return [labels]
    if not isinstance(labels, Sequence) or isinstance(labels, (bytes, bytearray)):
        return []
    if not labels:
        return []
    if all(isinstance(label, str) for label in labels):
        return [str(label) for label in labels]
    if index >= len(labels) or not isinstance(labels[index], Sequence):
        return []
    return [str(label) for label in labels[index]]


def _match_predictions(
    predictions: Sequence[ScoredDetection],
    targets: Sequence[tuple[float, float, float, float]],
    *,
    category: str | None,
    target_labels: Sequence[str] | None,
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    matches = []
    used_targets: set[int] = set()
    prediction_order = sorted(
        range(len(predictions)),
        key=lambda index: (-predictions[index].score, index),
    )
    for prediction_index in prediction_order:
        prediction = predictions[prediction_index]
        if category and prediction.label and prediction.label.casefold() != category:
            continue
        best_target = -1
        best_iou = iou_threshold
        for target_index, target in enumerate(targets):
            if target_index in used_targets:
                continue
            if (
                target_labels
                and target_index < len(target_labels)
                and target_labels[target_index]
                and prediction.label
                and prediction.label.casefold() != target_labels[target_index].casefold()
            ):
                continue
            iou = compute_iou(prediction.box, target)
            if iou >= best_iou:
                best_iou = iou
                best_target = target_index
        if best_target >= 0:
            used_targets.add(best_target)
            matches.append((prediction_index, best_target, best_iou))
    return matches


def _prediction_batch(completions: Sequence[object]) -> list[list[ScoredDetection]]:
    return [parse_reward_detections(completion_text(completion)) for completion in completions]


def iou_reward(
    completions: Sequence[object],
    target_boxes: object = None,
    category: object = None,
    iou_threshold: float = 0.5,
    solution: object = None,
    target_labels: object = None,
    **kwargs,
) -> list[float]:
    """Reward correctly localized detections with one-to-one matching."""
    del kwargs
    if target_boxes is None:
        target_boxes = solution
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    rewards = []
    for index, predictions in enumerate(_prediction_batch(completions)):
        targets = _target_for_index(target_boxes, index)
        matches = _match_predictions(
            predictions,
            targets,
            category=_category_for_index(category, index),
            target_labels=_labels_for_index(target_labels, index),
            iou_threshold=iou_threshold,
        )
        if not targets and not predictions:
            rewards.append(1.0)
        else:
            rewards.append(
                sum(match[2] for match in matches)
                / max(len(targets), len(predictions), 1)
            )
    return rewards


def confidence_reward(
    completions: Sequence[object],
    target_boxes: object = None,
    category: object = None,
    iou_threshold: float = 0.5,
    solution: object = None,
    target_labels: object = None,
    **kwargs,
) -> list[float]:
    """Reward high scores on matched boxes and low scores on false positives.

    This is the score constraint used by Visual-RFT.  A prediction matched to
    a target contributes its emitted score; an unmatched prediction
    contributes ``1 - score``.  The score is zero when a detection omits the
    explicit score field, which makes score-free SFT answers ineligible for
    this reward.
    """
    del kwargs
    if target_boxes is None:
        target_boxes = solution
    rewards = []
    for index, predictions in enumerate(_prediction_batch(completions)):
        targets = _target_for_index(target_boxes, index)
        matches = _match_predictions(
            predictions,
            targets,
            category=_category_for_index(category, index),
            target_labels=_labels_for_index(target_labels, index),
            iou_threshold=iou_threshold,
        )
        matched = {prediction_index for prediction_index, _, _ in matches}
        values = []
        for prediction_index, prediction in enumerate(predictions):
            if not prediction.has_explicit_score:
                values.append(0.0)
            elif prediction_index in matched:
                values.append(prediction.score)
            else:
                values.append(1.0 - prediction.score)
        if not targets and not predictions:
            rewards.append(1.0)
        elif not values:
            rewards.append(0.0)
        else:
            rewards.append(sum(values) / len(values))
    return rewards


def format_reward(completions: Sequence[object], **kwargs) -> list[float]:
    """Reward a parseable detection list whose entries explicitly have scores."""
    del kwargs
    rewards = []
    for completion in completions:
        text = completion_text(completion)
        parsed = parse_reward_detections(text)
        stripped = _answer_content(text)
        is_empty = stripped == "[]"
        rewards.append(
            1.0
            if (is_empty or parsed)
            and all(prediction.has_explicit_score for prediction in parsed)
            else 0.0
        )
    return rewards


def score_reward(completions: Sequence[object], **kwargs) -> list[float]:
    """Alias for the confidence reward used in the CLI registry."""
    return confidence_reward(completions, **kwargs)


def accuracy_reward_iou(completions, solution=None, **kwargs):
    """Visual-RFT-compatible alias accepting ``solution`` as target boxes."""
    if kwargs.get("target_boxes") is None and solution is not None:
        kwargs["target_boxes"] = solution
    return iou_reward(completions, **kwargs)


def accuracy_reward_confidence(completions, solution=None, **kwargs):
    """Visual-RFT-compatible alias for the score reward."""
    if kwargs.get("target_boxes") is None and solution is not None:
        kwargs["target_boxes"] = solution
    return confidence_reward(completions, **kwargs)


REWARD_FUNCS_REGISTRY = {
    "iou": iou_reward,
    "score": score_reward,
    "confidence": confidence_reward,
    "format": format_reward,
    "accuracy_iou": accuracy_reward_iou,
    "accuracy_confidence": accuracy_reward_confidence,
}

# The lower-case name mirrors the reference project and is convenient for
# callers that import the registry directly.
reward_funcs_registry = REWARD_FUNCS_REGISTRY


__all__ = [
    "REWARD_FUNCS_REGISTRY",
    "ScoredDetection",
    "accuracy_reward_confidence",
    "accuracy_reward_iou",
    "completion_text",
    "confidence_reward",
    "format_reward",
    "iou_reward",
    "parse_reward_detections",
    "parse_target_boxes",
    "reward_funcs_registry",
    "score_reward",
]
