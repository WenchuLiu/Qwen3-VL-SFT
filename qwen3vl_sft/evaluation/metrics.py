"""Dependency-light parsing and metrics for normalized detection outputs."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Iterable, Sequence


IOU_THRESHOLDS = tuple(0.50 + 0.05 * index for index in range(10))
_RECORD_PATTERN = re.compile(
    r"\(\s*(?P<label>[^,();]+?)\s*,\s*"
    r"(?P<x1>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"(?P<y1>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"(?P<x2>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"(?P<y2>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\)\s*;"
)


def _valid_box(values: Sequence[object]) -> list[float] | None:
    if len(values) != 4:
        return None
    try:
        box = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    if not all(0.0 <= value <= 1000.0 for value in box):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def parse_detection_output(text: str) -> list[tuple[str, list[float]]]:
    """Parse JSON grounding output, with compatibility for legacy records."""
    if not isinstance(text, str):
        return []

    legacy = []
    for match in _RECORD_PATTERN.finditer(text):
        label = match.group("label").strip().strip("\"'")
        box = _valid_box([match.group(name) for name in ("x1", "y1", "x2", "y2")])
        if label and box is not None:
            legacy.append((label, box))
    if legacy:
        return legacy

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        entries = candidate if isinstance(candidate, list) else [candidate]
        parsed: list[tuple[str, list[float]]] = []
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("label"), str):
                continue
            box = _valid_box(entry.get("bbox_2d", []))
            if box is not None and entry["label"].strip():
                parsed.append((entry["label"].strip(), box))
        # A valid JSON list with no usable detections means an intentional [];
        # return it immediately rather than accidentally parsing a later list.
        if isinstance(candidate, list):
            return parsed
        if parsed:
            return parsed
    return []


def compute_iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - intersection
    return intersection / union if union > 1e-12 else 0.0


def greedy_match(
    predictions: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    *,
    iou_threshold: float,
) -> tuple[int, int, int]:
    """Greedy one-to-one matching with deterministic tie breaking."""
    pairs = sorted(
        (
            -compute_iou(prediction, target),
            prediction_index,
            target_index,
        )
        for prediction_index, prediction in enumerate(predictions)
        for target_index, target in enumerate(targets)
        if compute_iou(prediction, target) >= iou_threshold
    )
    matched_predictions: set[int] = set()
    matched_targets: set[int] = set()
    for _, prediction_index, target_index in pairs:
        if prediction_index not in matched_predictions and target_index not in matched_targets:
            matched_predictions.add(prediction_index)
            matched_targets.add(target_index)
    true_positive = len(matched_predictions)
    return true_positive, len(predictions) - true_positive, len(targets) - len(matched_targets)


def _matched_iou(predictions: Sequence[Sequence[float]], targets: Sequence[Sequence[float]]) -> tuple[float, int]:
    pairs = sorted(
        (
            -compute_iou(prediction, target),
            prediction_index,
            target_index,
        )
        for prediction_index, prediction in enumerate(predictions)
        for target_index, target in enumerate(targets)
    )
    used_predictions: set[int] = set()
    used_targets: set[int] = set()
    values: list[float] = []
    for negative_iou, prediction_index, target_index in pairs:
        if prediction_index in used_predictions or target_index in used_targets:
            continue
        used_predictions.add(prediction_index)
        used_targets.add(target_index)
        values.append(-negative_iou)
    return sum(values), len(values)


def _empty_shot_metrics() -> dict:
    return {
        "episodes": 0,
        "count_exact": 0,
        "matched_iou_sum": 0.0,
        "matched_iou_count": 0,
        "thresholds": {
            f"IoU@{threshold:.2f}": {"tp": 0, "fp": 0, "fn": 0}
            for threshold in IOU_THRESHOLDS
        },
    }


def _finalize_shot_metrics(values: dict) -> dict:
    episode_count = values["episodes"]
    denominator = episode_count or 1
    thresholds = {}
    f1_values = []
    for key, counts in values["thresholds"].items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        thresholds[key] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        f1_values.append(f1)
    return {
        "episodes": episode_count,
        "count_accuracy": values["count_exact"] / denominator,
        "mean_matched_iou": (
            values["matched_iou_sum"] / values["matched_iou_count"]
            if values["matched_iou_count"] else 0.0
        ),
        "thresholds": thresholds,
        "f1_mean_over_iou": sum(f1_values) / len(f1_values),
    }


def evaluate_episode_predictions(
    episodes: Iterable[dict],
    responses: Iterable[str],
) -> dict:
    """Compute per-shot metrics and retain every raw response for auditing."""
    episodes = list(episodes)
    responses = list(responses)
    if len(episodes) != len(responses):
        raise ValueError(
            f"episode/response count mismatch: {len(episodes)} episodes, "
            f"{len(responses)} responses"
        )
    metrics_by_shot = defaultdict(_empty_shot_metrics)
    predictions = []
    for episode, response in zip(episodes, responses):
        category = episode["category"]
        target_boxes = episode["query"].get("boxes", [])
        predicted_boxes = [
            box for label, box in parse_detection_output(response) if label == category
        ]
        shot = str(episode["num_shots"])
        values = metrics_by_shot[shot]
        values["episodes"] += 1
        values["count_exact"] += int(len(predicted_boxes) == len(target_boxes))
        iou_sum, iou_count = _matched_iou(predicted_boxes, target_boxes)
        values["matched_iou_sum"] += iou_sum
        values["matched_iou_count"] += iou_count
        for threshold in IOU_THRESHOLDS:
            counts = values["thresholds"][f"IoU@{threshold:.2f}"]
            tp, fp, fn = greedy_match(
                predicted_boxes, target_boxes, iou_threshold=threshold
            )
            counts["tp"] += tp
            counts["fp"] += fp
            counts["fn"] += fn
        predictions.append(
            {
                "id": episode.get("id"),
                "category": category,
                "num_shots": episode["num_shots"],
                "query_image_id": episode["query"].get("image_id"),
                "target_boxes": target_boxes,
                "predicted_boxes": predicted_boxes,
                "response": response,
            }
        )
    return {
        "metrics_by_shot": {
            shot: _finalize_shot_metrics(values)
            for shot, values in sorted(metrics_by_shot.items(), key=lambda item: int(item[0]))
        },
        "predictions": predictions,
    }


def trainer_metrics(result: dict) -> dict[str, float]:
    """Flatten the shared result into Hugging Face Trainer metric keys."""
    metrics: dict[str, float] = {}
    by_shot = result["metrics_by_shot"]
    for shot, summary in by_shot.items():
        prefix = f"eval_coco_{shot}shot"
        metrics[f"{prefix}_f1"] = float(summary["f1_mean_over_iou"])
        metrics[f"{prefix}_count_accuracy"] = float(summary["count_accuracy"])
        metrics[f"{prefix}_mean_matched_iou"] = float(summary["mean_matched_iou"])
        for threshold, values in summary["thresholds"].items():
            suffix = threshold.lower().replace("@", "_").replace(".", "_")
            metrics[f"{prefix}_{suffix}_f1"] = float(values["f1"])
    if "1" in by_shot:
        one_shot = by_shot["1"]
        metrics["eval_coco_f1"] = float(one_shot["f1_mean_over_iou"])
        metrics["eval_coco_count_accuracy"] = float(one_shot["count_accuracy"])
        metrics["eval_coco_mean_matched_iou"] = float(one_shot["mean_matched_iou"])
        metrics["eval_coco_iou_0_50_f1"] = float(one_shot["thresholds"]["IoU@0.50"]["f1"])
    metrics["eval_coco_episodes"] = float(
        sum(summary["episodes"] for summary in by_shot.values())
    )
    return metrics
