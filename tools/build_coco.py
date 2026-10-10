#!/usr/bin/env python3
"""Build deterministic COCO SFT data or fixed generation episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen3vl_sft.evaluation.coco.data import (
    build_flexible_episode_records,
    build_eval_records,
    build_train_records,
    load_coco_frames,
    parse_shots,
    read_image_ids,
    select_image_ids,
    write_json,
)
from qwen3vl_sft.evaluation.coco.protocol import (
    FLEXIBLE_EPISODE_PROTOCOL,
    FLEXIBLE_EPISODE_TEMPLATE_VERSION,
    LOSS_MODE,
    PROTOCOL_NAME,
    PROMPT_TEMPLATE_VERSION,
)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", nargs="+", default=["1"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-box-area-ratio", type=float, default=0.001)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="build SFT conversation records")
    _common(train)
    train.add_argument("--num-samples", type=int, default=100000)
    train.add_argument("--image-fraction", type=float, default=1.0)
    train.add_argument("--image-ids-output", type=Path, default=None)
    train.add_argument("--all-selected-queries", action="store_true")
    train.add_argument("--skip-image-check", action="store_true")

    evaluation = subparsers.add_parser("eval", help="build fixed support/query episodes")
    evaluation.add_argument("--support-annotations", type=Path, required=True)
    evaluation.add_argument("--support-image-root", type=Path, required=True)
    evaluation.add_argument("--support-image-ids", type=Path, default=None)
    evaluation.add_argument("--query-annotations", type=Path, required=True)
    evaluation.add_argument("--query-image-root", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    query_selection = evaluation.add_mutually_exclusive_group()
    query_selection.add_argument(
        "--num-query-images",
        type=int,
        default=None,
        help="Number of unique query images (default: 500).",
    )
    query_selection.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of category-balanced episodes instead of unique query images.",
    )
    evaluation.add_argument("--shots", nargs="+", default=["1"])
    evaluation.add_argument("--seed", type=int, default=43)
    evaluation.add_argument("--min-box-area-ratio", type=float, default=0.001)
    evaluation.add_argument(
        "--all-query-pairs",
        action="store_true",
        help="Build one episode for every query image/category pair.",
    )

    episodes = subparsers.add_parser(
        "episodes",
        help="build flexible standard/support-defined/distractor training episodes",
    )
    episodes.add_argument("--annotations", type=Path, required=True)
    episodes.add_argument("--image-root", type=Path, required=True)
    episodes.add_argument(
        "--image-ids",
        type=Path,
        required=True,
        help="10% image-ID manifest used by the existing COCO training split",
    )
    episodes.add_argument(
        "--image-pool-seed",
        type=int,
        default=42,
        help="Expected seed recorded in the 10% image-ID manifest.",
    )
    episodes.add_argument(
        "--upper-category-map",
        type=Path,
        default=ROOT / "docs/coco_upper_category_map.json",
    )
    episodes.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/coco/train_episodes_10pct_flexible_11820.json",
    )
    episodes.add_argument("--num-episodes", type=int, default=11820)
    episodes.add_argument("--seed", type=int, default=42)
    episodes.add_argument("--min-box-area-ratio", type=float, default=0.001)
    episodes.add_argument("--distractor-category", default="dog")

    image_ids = subparsers.add_parser(
        "image-ids", help="write a deterministic image-ID manifest"
    )
    image_ids.add_argument("--annotations", type=Path, required=True)
    image_ids.add_argument("--output", type=Path, required=True)
    image_ids.add_argument("--fraction", type=float, default=0.1)
    image_ids.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_train(args: argparse.Namespace) -> None:
    if not args.annotations.is_file():
        raise FileNotFoundError(args.annotations)
    if not args.image_root.is_dir():
        raise NotADirectoryError(args.image_root)
    selected_ids = select_image_ids(args.annotations, args.image_fraction, args.seed)
    if args.image_ids_output:
        write_json(
            args.image_ids_output,
            {
                "fraction": args.image_fraction,
                "seed": args.seed,
                "image_ids": selected_ids,
            },
        )
    frames = load_coco_frames(
        args.annotations,
        args.image_root,
        min_box_area_ratio=args.min_box_area_ratio,
        verify_images=not args.skip_image_check,
        image_ids=selected_ids,
    )
    records = build_train_records(
        frames,
        shots=parse_shots(args.shots, allow_zero=True),
        num_samples=args.num_samples,
        seed=args.seed,
        all_selected_queries=args.all_selected_queries,
    )
    write_json(
        args.output,
        {
            "schema": "qwen-vl-conversations",
            "protocol": PROTOCOL_NAME,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "loss_mode": LOSS_MODE,
            "source_annotations": str(args.annotations.resolve()),
            "seed": args.seed,
            "records": records,
        },
    )
    print(f"Wrote {len(records)} SFT records to {args.output}")


def build_eval(args: argparse.Namespace) -> None:
    for path in (args.support_annotations, args.query_annotations):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (args.support_image_root, args.query_image_root):
        if not path.is_dir():
            raise NotADirectoryError(path)
    support_ids = read_image_ids(args.support_image_ids) if args.support_image_ids else None
    support_frames = load_coco_frames(
        args.support_annotations,
        args.support_image_root,
        min_box_area_ratio=args.min_box_area_ratio,
        image_ids=support_ids,
    )
    query_frames = load_coco_frames(
        args.query_annotations,
        args.query_image_root,
        min_box_area_ratio=args.min_box_area_ratio,
    )
    shots = parse_shots(args.shots, allow_zero=True)
    num_query_images = (
        500
        if args.num_query_images is None and args.num_samples is None
        else args.num_query_images
    )
    records = build_eval_records(
        support_frames,
        query_frames,
        shots=shots,
        seed=args.seed,
        num_query_images=num_query_images,
        num_samples=args.num_samples,
        all_query_pairs=args.all_query_pairs,
    )
    write_json(
        args.output,
        {
            "protocol": PROTOCOL_NAME,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "support_annotations": str(args.support_annotations.resolve()),
            "query_annotations": str(args.query_annotations.resolve()),
            "shots": shots,
            "seed": args.seed,
            "num_query_images": num_query_images,
            "num_samples": args.num_samples,
            "records": records,
        },
    )
    print(f"Wrote {len(records)} fixed evaluation episodes to {args.output}")


def build_episodes(args: argparse.Namespace) -> None:
    for path in (args.annotations, args.image_ids, args.upper_category_map):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.image_root.is_dir():
        raise NotADirectoryError(args.image_root)

    with args.image_ids.open("r", encoding="utf-8") as handle:
        image_manifest = json.load(handle)
    if (
        not isinstance(image_manifest, dict)
        or float(image_manifest.get("fraction", -1)) != 0.1
    ):
        raise ValueError(
            "--image-ids must be the existing 10% COCO image-ID manifest "
            "with fraction: 0.1"
        )
    try:
        manifest_seed = int(image_manifest["seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("10% image-ID manifest must include its integer seed") from error
    if manifest_seed != args.image_pool_seed:
        raise ValueError(
            f"image-ID manifest seed is {manifest_seed}, expected {args.image_pool_seed}"
        )
    selected_ids = read_image_ids(args.image_ids)

    with args.upper_category_map.open("r", encoding="utf-8") as handle:
        upper_map_payload = json.load(handle)
    if isinstance(upper_map_payload, dict) and "groups" in upper_map_payload:
        upper_category_map = upper_map_payload["groups"]
    else:
        upper_category_map = upper_map_payload
    if not isinstance(upper_category_map, dict):
        raise ValueError("upper-category map must be a JSON object of groups")

    frames = load_coco_frames(
        args.annotations,
        args.image_root,
        min_box_area_ratio=args.min_box_area_ratio,
        image_ids=selected_ids,
    )
    records = build_flexible_episode_records(
        frames,
        upper_category_map=upper_category_map,
        num_episodes=args.num_episodes,
        seed=args.seed,
        distractor_category=args.distractor_category,
    )
    counts_by_type: dict[str, int] = {}
    counts_by_shot: dict[str, int] = {}
    for record in records:
        episode_type = str(record["episode_type"])
        shot = str(record["num_shots"])
        counts_by_type[episode_type] = counts_by_type.get(episode_type, 0) + 1
        counts_by_shot[shot] = counts_by_shot.get(shot, 0) + 1
    write_json(
        args.output,
        {
            "schema": "coco-flexible-detection-episodes-v1",
            "protocol": FLEXIBLE_EPISODE_PROTOCOL,
            "prompt_template_version": FLEXIBLE_EPISODE_TEMPLATE_VERSION,
            "source_annotations": str(args.annotations.resolve()),
            "image_ids_manifest": str(args.image_ids.resolve()),
            "image_pool_size": len(selected_ids),
            "image_pool_fraction": 0.1,
            "image_pool_seed": manifest_seed,
            "upper_category_map_file": str(args.upper_category_map.resolve()),
            "upper_category_map": upper_category_map,
            "seed": args.seed,
            "num_episodes": len(records),
            "episode_type_counts": counts_by_type,
            "shot_counts": counts_by_shot,
            "records": records,
        },
    )
    print(
        f"Wrote {len(records)} flexible episodes from the 10% pool "
        f"({counts_by_type}) to {args.output}"
    )


def build_image_ids(args: argparse.Namespace) -> None:
    if not args.annotations.is_file():
        raise FileNotFoundError(args.annotations)
    selected_ids = select_image_ids(args.annotations, args.fraction, args.seed)
    write_json(
        args.output,
        {
            "fraction": args.fraction,
            "seed": args.seed,
            "image_ids": selected_ids,
        },
    )
    print(
        f"Wrote {len(selected_ids)} image IDs ({args.fraction:.0%}) to {args.output}"
    )


def main() -> None:
    args = parse_args()
    if args.command == "train":
        build_train(args)
    elif args.command == "eval":
        build_eval(args)
    elif args.command == "episodes":
        build_episodes(args)
    else:
        build_image_ids(args)


if __name__ == "__main__":
    main()
