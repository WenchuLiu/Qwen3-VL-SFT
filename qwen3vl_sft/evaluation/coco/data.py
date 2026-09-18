"""COCO annotation loading and deterministic support/query episode sampling."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .protocol import PROTOCOL_NAME, PROMPT_TEMPLATE_VERSION, build_sft_record


@dataclass(frozen=True)
class CocoFrame:
    # COCO-compatible datasets are allowed to use string image IDs.  ArTaxOr,
    # for example, uses hashes while the other local datasets use integers.
    image_id: int | str
    category_id: int
    category: str
    image_path: str
    boxes: tuple[tuple[int, int, int, int], ...]
    width: int = 1000
    height: int = 1000


def _identifier(value: object) -> int | str:
    """Keep COCO identifiers stable while accepting numeric JSON values."""
    if isinstance(value, bool):
        raise ValueError("boolean values are not valid COCO identifiers")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"unsupported COCO identifier: {value!r}")


def _identifier_sort_key(value: int | str) -> tuple[int, object]:
    if isinstance(value, int):
        return (0, value)
    return (1, value)


def _xywh_to_normalized(
    bbox: Sequence[float], width: int, height: int
) -> tuple[int, int, int, int] | None:
    if width <= 0 or height <= 0:
        return None
    try:
        if len(bbox) != 4:
            return None
        x, y, box_width, box_height = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    left = max(0.0, min(float(width), x))
    top = max(0.0, min(float(height), y))
    right = max(0.0, min(float(width), x + box_width))
    bottom = max(0.0, min(float(height), y + box_height))
    if right <= left or bottom <= top:
        return None
    return (
        round(left / width * 1000),
        round(top / height * 1000),
        round(right / width * 1000),
        round(bottom / height * 1000),
    )


def load_coco_frames(
    annotations_path: str | Path,
    image_root: str | Path,
    *,
    min_box_area_ratio: float = 0.001,
    verify_images: bool = True,
    image_ids: Iterable[int | str] | None = None,
) -> dict[str, list[CocoFrame]]:
    """Group usable COCO boxes by category and image.

    Each frame keeps normalized 0-1000 boxes, which is the coordinate system
    used both in SFT targets and in generation-time matching.
    """
    if not 0 <= min_box_area_ratio < 1:
        raise ValueError("min_box_area_ratio must be in [0, 1)")
    annotation_path = Path(annotations_path)
    image_root = Path(image_root)
    with annotation_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)
    if not isinstance(coco, dict):
        raise ValueError(f"COCO annotations must be a JSON object: {annotation_path}")

    selected_ids = (
        {_identifier(value) for value in image_ids} if image_ids is not None else None
    )
    images = {
        _identifier(image["id"]): image
        for image in coco.get("images", [])
    }
    categories = {
        int(category["id"]): str(category["name"])
        for category in coco.get("categories", [])
    }
    grouped: dict[tuple[int | str, int], list[tuple[int, int, int, int]]] = defaultdict(list)

    for annotation in coco.get("annotations", []):
        if annotation.get("iscrowd", 0):
            continue
        try:
            image_id = _identifier(annotation["image_id"])
            category_id = int(annotation["category_id"])
        except (KeyError, TypeError, ValueError):
            continue
        image = images.get(image_id)
        category = categories.get(category_id)
        if (
            image is None
            or category is None
            or (selected_ids is not None and image_id not in selected_ids)
        ):
            continue
        try:
            width, height = int(image["width"]), int(image["height"])
            bbox = annotation["bbox"]
            area_ratio = float(bbox[2]) * float(bbox[3]) / (width * height)
        except (KeyError, TypeError, ValueError, ZeroDivisionError, IndexError):
            continue
        if area_ratio < min_box_area_ratio:
            continue
        image_name = Path(str(image["file_name"]))
        image_path = image_name if image_name.is_absolute() else image_root / image_name
        if verify_images and not image_path.is_file():
            continue
        normalized = _xywh_to_normalized(bbox, width, height)
        if normalized is not None:
            grouped[(image_id, category_id)].append(normalized)

    result: dict[str, list[CocoFrame]] = defaultdict(list)
    for (image_id, category_id), boxes in grouped.items():
        image = images[image_id]
        image_name = Path(str(image["file_name"]))
        image_path = image_name if image_name.is_absolute() else image_root / image_name
        result[categories[category_id]].append(
            CocoFrame(
                image_id=image_id,
                category_id=category_id,
                category=categories[category_id],
                image_path=str(image_path.resolve()),
                boxes=tuple(sorted(set(boxes))),
                width=int(image["width"]),
                height=int(image["height"]),
            )
        )
    for frames in result.values():
        frames.sort(key=lambda frame: _identifier_sort_key(frame.image_id))
    return dict(result)


def select_image_ids(
    annotations_path: str | Path, fraction: float, seed: int
) -> list[int | str]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    with Path(annotations_path).open("r", encoding="utf-8") as handle:
        coco = json.load(handle)
    image_ids = [
        _identifier(image["id"])
        for image in coco.get("images", [])
    ]
    image_ids.sort(key=_identifier_sort_key)
    if fraction == 1:
        return image_ids
    count = max(1, round(len(image_ids) * fraction))
    return sorted(random.Random(seed).sample(image_ids, count))


def read_image_ids(path: str | Path) -> set[int | str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("image_ids")
    if not isinstance(payload, list):
        raise ValueError(f"expected a list of image IDs in {path}")
    return {_identifier(value) for value in payload}


def parse_shots(values: Sequence[str], *, allow_zero: bool = False) -> list[int]:
    shots: list[int] = []
    for value in values:
        for item in value.split(","):
            shot = int(item)
            if shot < 0 or (shot == 0 and not allow_zero):
                raise ValueError(
                    "shot counts must be positive"
                    if not allow_zero
                    else "shot counts must be non-negative"
                )
            shots.append(shot)
    if not shots:
        raise ValueError("at least one shot count is required")
    return sorted(set(shots))


def _frame_record(frame: CocoFrame) -> dict:
    return {
        "image_id": frame.image_id,
        "category_id": frame.category_id,
        "image": frame.image_path,
        "width": frame.width,
        "height": frame.height,
        "boxes": [list(box) for box in frame.boxes],
    }


def _sample_support(
    frames: Sequence[CocoFrame], query: CocoFrame, shots: int, rng: random.Random
) -> list[CocoFrame]:
    candidates = [frame for frame in frames if frame.image_id != query.image_id]
    if len(candidates) < shots:
        raise ValueError(
            f"category {query.category!r} has {len(candidates)} support frames, "
            f"but {shots} are required"
        )
    return rng.sample(candidates, shots)


def build_train_records(
    frames_by_category: Mapping[str, Sequence[CocoFrame]],
    *,
    shots: Sequence[int],
    num_samples: int | None,
    seed: int,
    all_selected_queries: bool = False,
) -> list[dict]:
    """Build class-balanced, deterministic SFT episodes."""
    if not shots or any(shot < 1 for shot in shots):
        raise ValueError("SFT shot counts must be positive")
    if not all_selected_queries and (num_samples is None or num_samples < 1):
        raise ValueError("num_samples must be positive")
    largest_shot = max(shots)
    categories = sorted(
        category for category, frames in frames_by_category.items()
        if len(frames) >= largest_shot + 1
    )
    if not categories:
        raise ValueError(f"no category has at least {largest_shot + 1} usable images")

    rng = random.Random(seed)
    candidates: list[tuple[str, CocoFrame]]
    if all_selected_queries:
        candidates = [
            (category, frame)
            for category in categories
            for frame in frames_by_category[category]
        ]
        rng.shuffle(candidates)
    else:
        assert num_samples is not None
        category_schedule = [categories[index % len(categories)] for index in range(num_samples)]
        rng.shuffle(category_schedule)
        candidates = [
            (category, rng.choice(list(frames_by_category[category])))
            for category in category_schedule
        ]

    records = []
    for index, (category, query) in enumerate(candidates):
        shot = shots[index % len(shots)]
        support = _sample_support(frames_by_category[category], query, shot, rng)
        records.append(
            build_sft_record(
                record_id=f"coco_sft_{index:07d}",
                category=category,
                support=[_frame_record(frame) for frame in support],
                query=_frame_record(query),
            )
        )
    return records


def build_eval_records(
    support_frames: Mapping[str, Sequence[CocoFrame]],
    query_frames: Mapping[str, Sequence[CocoFrame]],
    *,
    shots: Sequence[int],
    seed: int,
    num_query_images: int | None = 500,
    num_samples: int | None = None,
    all_query_pairs: bool = False,
) -> list[dict]:
    """Build fixed, support/query-disjoint evaluation episodes."""
    if not shots or any(shot < 0 for shot in shots):
        raise ValueError("evaluation shot counts must be non-negative")
    if not all_query_pairs and num_query_images is not None and num_query_images < 1:
        raise ValueError("num_query_images must be positive")
    if (
        not all_query_pairs
        and num_query_images is None
        and (num_samples is None or num_samples < 1)
    ):
        raise ValueError("num_samples must be positive when num_query_images is unset")

    categories = sorted(
        category for category in set(support_frames) & set(query_frames)
        if (
            query_frames[category]
            and (
                len(support_frames[category]) >= max(shots)
                or (all_query_pairs and support_frames[category])
            )
        )
    )
    if not categories:
        raise ValueError("no category has enough support and query frames")

    rng = random.Random(seed)
    records: list[dict] = []
    if all_query_pairs:
        candidates = [
            (category, frame)
            for category in categories
            for frame in query_frames[category]
        ]
        for index, (category, query) in enumerate(candidates):
            shot = shots[index % len(shots)]
            support_count = min(shot, len(support_frames[category]))
            support = _sample_support(
                support_frames[category],
                query,
                support_count,
                rng,
            )
            records.append(
                {
                    "id": f"coco_eval_{index:07d}",
                    "protocol": PROTOCOL_NAME,
                    "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                    "category": category,
                    "num_shots": shot,
                    "support": [_frame_record(frame) for frame in support],
                    "query": _frame_record(query),
                }
            )
        return records

    if num_query_images is not None:
        options_by_image: dict[int | str, list[tuple[str, CocoFrame]]] = defaultdict(list)
        for category in categories:
            for frame in query_frames[category]:
                options_by_image[frame.image_id].append((category, frame))
        available = sorted(options_by_image)
        if num_query_images > len(available):
            raise ValueError(
                f"requested {num_query_images} query images, but only "
                f"{len(available)} are available"
            )
        selected = rng.sample(available, num_query_images)
        for index, image_id in enumerate(selected):
            category, query = rng.choice(options_by_image[image_id])
            shot = shots[index % len(shots)]
            support = _sample_support(support_frames[category], query, shot, rng)
            records.append(
                {
                    "id": f"coco_eval_{index:07d}",
                    "protocol": PROTOCOL_NAME,
                    "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                    "category": category,
                    "num_shots": shot,
                    "support": [_frame_record(frame) for frame in support],
                    "query": _frame_record(query),
                }
            )
        return records

    assert num_samples is not None
    category_schedule = [categories[index % len(categories)] for index in range(num_samples)]
    rng.shuffle(category_schedule)
    for index, category in enumerate(category_schedule):
        query = rng.choice(list(query_frames[category]))
        shot = shots[index % len(shots)]
        support = _sample_support(support_frames[category], query, shot, rng)
        records.append(
            {
                "id": f"coco_eval_{index:07d}",
                "protocol": PROTOCOL_NAME,
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "category": category,
                "num_shots": shot,
                "support": [_frame_record(frame) for frame in support],
                "query": _frame_record(query),
            }
        )
    return records


def build_fixed_support_eval_records(
    support_frames: Mapping[str, Sequence[CocoFrame]],
    query_frames: Mapping[str, Sequence[CocoFrame]],
    *,
    shot: int,
) -> list[dict]:
    """Build class-wise positive episodes from a prepared support file.

    The local few-shot datasets already provide fixed ``1_shot.json``,
    ``2_shot.json`` and ``4_shot.json`` support annotations.  Preserve their
    order and use every available support image up to ``shot``.  Some source
    files contain multiple boxes from one image, so a nominal 4-shot file can
    have only three distinct support images for a category; retaining that
    category matches the original baseline's behavior instead of silently
    dropping query images.
    """
    if shot < 0:
        raise ValueError("shot count must be non-negative")
    categories = sorted(
        category
        for category in set(support_frames) & set(query_frames)
        if support_frames[category] or shot == 0
    )
    if not categories:
        raise ValueError("no category has support and query frames")

    records: list[dict] = []
    record_index = 0
    for category in categories:
        support = list(support_frames.get(category, ()))[:shot]
        for query in query_frames[category]:
            records.append(
                {
                    "id": f"coco_eval_{record_index:07d}",
                    "protocol": PROTOCOL_NAME,
                    "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                    "category": category,
                    "num_shots": shot,
                    "support": [_frame_record(frame) for frame in support],
                    "query": _frame_record(query),
                }
            )
            record_index += 1
    return records


def write_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

