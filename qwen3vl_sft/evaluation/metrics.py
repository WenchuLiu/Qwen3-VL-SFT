"""Dependency-light parsing and metrics for normalized detection outputs."""

from __future__ import annotations

import io
import json
import math
import re
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path
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


def _valid_score(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.5
    if not math.isfinite(score) or score == -1.0:
        return 0.5
    return min(1.0, max(0.0, score))


def parse_scored_detection_output(
    text: str,
) -> list[tuple[str, list[float], float]]:
    """Parse DetPO-style grounding output and retain per-box model scores."""
    if not isinstance(text, str):
        return []

    legacy: list[tuple[str, list[float], float]] = []
    for match in _RECORD_PATTERN.finditer(text):
        label = match.group("label").strip().strip("\"'")
        box = _valid_box([match.group(name) for name in ("x1", "y1", "x2", "y2")])
        if label and box is not None:
            legacy.append((label, box, 0.5))
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
        parsed: list[tuple[str, list[float], float]] = []
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("label"), str):
                continue
            box = _valid_box(entry.get("bbox_2d", []))
            if box is not None and entry["label"].strip():
                parsed.append(
                    (entry["label"].strip(), box, _valid_score(entry.get("score", 0.5)))
                )
        # A valid JSON list with no usable detections means an intentional [];
        # return it immediately rather than accidentally parsing a later list.
        if isinstance(candidate, list):
            return parsed
        if parsed:
            return parsed
    return []


def parse_detection_output(text: str) -> list[tuple[str, list[float]]]:
    """Parse boxes while preserving the pre-mAP public return shape."""
    return [(label, box) for label, box, _ in parse_scored_detection_output(text)]


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


COCO_STAT_NAMES = (
    "map_50_95",
    "map_50",
    "map_75",
    "map_small",
    "map_medium",
    "map_large",
    "ar_1",
    "ar_10",
    "ar_100",
    "ar_small",
    "ar_medium",
    "ar_large",
)


def _empty_coco_metrics() -> dict[str, object]:
    return {"stats": [0.0] * len(COCO_STAT_NAMES), **dict.fromkeys(COCO_STAT_NAMES, 0.0)}


def _synthetic_coco_metrics(
    episodes: Sequence[dict],
    responses: Sequence[str],
    *,
    score_mode: str,
) -> dict[str, object]:
    """Compatibility metric for in-memory records without COCO provenance.

    Real CLI and Trainer evaluations always provide ``query_annotations`` and
    use the official branch below.  This fallback keeps small protocol tests
    and legacy callers, which have no original COCO IDs/annotation file,
    evaluable with the same pycocotools implementation.
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:
        raise ImportError(
            "generation mAP requires pycocotools; install the project dependencies"
        ) from error

    categories = sorted({str(episode["category"]) for episode in episodes})
    category_ids = {category: index + 1 for index, category in enumerate(categories)}
    images = []
    annotations = []
    detections = []
    annotation_id = 1

    def to_pixels(box: Sequence[float], width: int, height: int) -> list[float]:
        x1, y1, x2, y2 = (float(value) for value in box)
        return [
            x1 / 1000.0 * width,
            y1 / 1000.0 * height,
            x2 / 1000.0 * width,
            y2 / 1000.0 * height,
        ]

    for image_id, (episode, response) in enumerate(zip(episodes, responses), start=1):
        category = str(episode["category"])
        category_id = category_ids[category]
        query = episode["query"]
        width = int(query.get("width", 1000))
        height = int(query.get("height", 1000))
        images.append({"id": image_id, "width": width, "height": height})
        for box in query.get("boxes", []):
            x1, y1, x2, y2 = to_pixels(box, width, height)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
        parsed = [
            (box, score)
            for label, box, score in parse_scored_detection_output(response)
            if label == category
        ]
        count = len(parsed)
        for rank, (box, model_score) in enumerate(parsed):
            x1, y1, x2, y2 = to_pixels(box, width, height)
            score = model_score if score_mode == "model" else (
                1.0 if count == 1 else 1.0 - 0.9 * rank / (count - 1)
            )
            detections.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score,
                }
            )
    if not annotations or not detections:
        return _empty_coco_metrics()
    coco_gt = COCO()
    coco_gt.dataset = {
        "info": {},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": category_id, "name": category}
            for category, category_id in category_ids.items()
        ],
    }
    with redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(detections)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.imgIds = [image["id"] for image in images]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = [float(value) for value in evaluator.stats]
    return {"stats": stats, **dict(zip(COCO_STAT_NAMES, stats))}


def _coco_metrics(
    episodes: Sequence[dict],
    responses: Sequence[str],
    *,
    score_mode: str,
    coco_annotations_path: str | None,
) -> dict[str, object]:
    """Evaluate task-conditioned detections with the official COCO API.

    DetPO loads the dataset's COCO annotation file and feeds predictions with
    the dataset's original image/category IDs to ``COCOeval``.  An ICL episode
    asks for one category in one query image, so construct a task subset of
    that official ground truth rather than treating normalized episode boxes as
    a new synthetic dataset.
    """
    if not coco_annotations_path:
        return _synthetic_coco_metrics(episodes, responses, score_mode=score_mode)
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:
        raise ImportError(
            "generation mAP requires pycocotools; install the project dependencies"
        ) from error

    annotation_path = Path(coco_annotations_path).expanduser().resolve()
    if not annotation_path.is_file():
        raise FileNotFoundError(f"COCO query annotations not found: {annotation_path}")

    coco_gt = COCO(str(annotation_path))
    task_pairs: set[tuple[int, int]] = set()
    task_image_ids: set[int] = set()
    task_category_ids: set[int] = set()
    for episode in episodes:
        query = episode["query"]
        try:
            image_id = int(query["image_id"])
            category_id = int(query["category_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "official COCO mAP requires query.image_id and query.category_id"
            ) from error
        if image_id not in coco_gt.imgs:
            raise ValueError(f"query image_id {image_id} is absent from {annotation_path}")
        if category_id not in coco_gt.cats:
            raise ValueError(f"query category_id {category_id} is absent from {annotation_path}")
        task_pairs.add((image_id, category_id))
        task_image_ids.add(image_id)
        task_category_ids.add(category_id)

    images = [coco_gt.imgs[image_id] for image_id in sorted(task_image_ids)]
    annotations = [
        annotation
        for annotation in coco_gt.dataset.get("annotations", [])
        if (int(annotation["image_id"]), int(annotation["category_id"])) in task_pairs
    ]
    categories = [coco_gt.cats[category_id] for category_id in sorted(task_category_ids)]
    detections = []

    def to_pixels(box: Sequence[float], width: int, height: int) -> list[int]:
        x1, y1, x2, y2 = (float(value) for value in box)
        return [
            int(x1 / 1000.0 * width),
            int(y1 / 1000.0 * height),
            int(x2 / 1000.0 * width),
            int(y2 / 1000.0 * height),
        ]

    for episode, response in zip(episodes, responses):
        category = str(episode["category"])
        query = episode["query"]
        image_id = int(query["image_id"])
        category_id = int(query["category_id"])
        width = int(query.get("width", 1000))
        height = int(query.get("height", 1000))
        if width < 1 or height < 1:
            raise ValueError(f"episode {episode.get('id', image_id)!r} has invalid image size")
        parsed = [
            (box, score)
            for label, box, score in parse_scored_detection_output(response)
            if label == category
        ]
        count = len(parsed)
        for rank, (box, model_score) in enumerate(parsed):
            x1, y1, x2, y2 = to_pixels(box, width, height)
            score = model_score
            if score_mode == "ranking":
                score = 1.0 if count == 1 else 1.0 - 0.9 * rank / (count - 1)
            detections.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score,
                }
            )

    if not annotations or not detections:
        return _empty_coco_metrics()

    coco_gt_subset = COCO()
    coco_gt_subset.dataset = {
        "info": coco_gt.dataset.get("info", {}),
        "licenses": coco_gt.dataset.get("licenses", []),
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    # pycocotools is intentionally chatty; result JSON and CLI summaries are
    # the stable reporting surface for this project.
    with redirect_stdout(io.StringIO()):
        coco_gt_subset.createIndex()
        coco_dt = coco_gt_subset.loadRes(detections)
        evaluator = COCOeval(coco_gt_subset, coco_dt, "bbox")
        evaluator.params.imgIds = sorted(task_image_ids)
        evaluator.params.catIds = sorted(task_category_ids)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = [float(value) for value in evaluator.stats]
    return {"stats": stats, **dict(zip(COCO_STAT_NAMES, stats))}


def evaluate_episode_predictions(
    episodes: Iterable[dict],
    responses: Iterable[str],
    *,
    coco_annotations_path: str | None = None,
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
        scored_predictions = [
            (box, score)
            for label, box, score in parse_scored_detection_output(response)
            if label == category
        ]
        predicted_boxes = [box for box, _ in scored_predictions]
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
                "predicted_detections": [
                    {"bbox_2d": box, "score": score} for box, score in scored_predictions
                ],
                "response": response,
            }
        )
    finalized = {
        shot: _finalize_shot_metrics(values)
        for shot, values in sorted(metrics_by_shot.items(), key=lambda item: int(item[0]))
    }
    for shot, summary in finalized.items():
        selected = [
            (episode, response)
            for episode, response in zip(episodes, responses)
            if str(episode["num_shots"]) == shot
        ]
        shot_episodes = [item[0] for item in selected]
        shot_responses = [item[1] for item in selected]
        summary["coco_map"] = {
            "model": _coco_metrics(
                shot_episodes,
                shot_responses,
                score_mode="model",
                coco_annotations_path=coco_annotations_path,
            ),
            "ranking": _coco_metrics(
                shot_episodes,
                shot_responses,
                score_mode="ranking",
                coco_annotations_path=coco_annotations_path,
            ),
        }
    return {
        "metrics_by_shot": finalized,
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
        metrics[f"{prefix}_map"] = float(summary["coco_map"]["model"]["map_50_95"])
        metrics[f"{prefix}_map_50"] = float(summary["coco_map"]["model"]["map_50"])
        metrics[f"{prefix}_map_75"] = float(summary["coco_map"]["model"]["map_75"])
        metrics[f"{prefix}_ranking_map"] = float(
            summary["coco_map"]["ranking"]["map_50_95"]
        )
        for threshold, values in summary["thresholds"].items():
            suffix = threshold.lower().replace("@", "_").replace(".", "_")
            metrics[f"{prefix}_{suffix}_f1"] = float(values["f1"])
    if "1" in by_shot:
        one_shot = by_shot["1"]
        metrics["eval_coco_f1"] = float(one_shot["f1_mean_over_iou"])
        metrics["eval_coco_count_accuracy"] = float(one_shot["count_accuracy"])
        metrics["eval_coco_mean_matched_iou"] = float(one_shot["mean_matched_iou"])
        metrics["eval_coco_iou_0_50_f1"] = float(one_shot["thresholds"]["IoU@0.50"]["f1"])
        metrics["eval_coco_map"] = float(one_shot["coco_map"]["model"]["map_50_95"])
        metrics["eval_coco_map_50"] = float(one_shot["coco_map"]["model"]["map_50"])
        metrics["eval_coco_map_75"] = float(one_shot["coco_map"]["model"]["map_75"])
    metrics["eval_coco_episodes"] = float(
        sum(summary["episodes"] for summary in by_shot.values())
    )
    return metrics
