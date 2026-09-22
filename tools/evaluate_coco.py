#!/usr/bin/env python3
"""Evaluate a base model or LoRA adapter on fixed COCO ICL episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen3vl_sft.config import DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS
from qwen3vl_sft.evaluation.coco.generation import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--episodes", required=True)
    parser.add_argument(
        "--data-root",
        default=None,
        help="Root containing COCO/train2017, COCO/val2017, and annotations.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--attention",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )
    parser.add_argument(
        "--ve",
        "--visual-enhancement",
        dest="visual_enhancement",
        action="store_true",
        help="Draw red ground-truth boxes on support images only.",
    )
    parser.add_argument(
        "--ie",
        "--instruction-enhancement",
        dest="instruction_enhancement",
        action="store_true",
        help="Add the requested category description to support and query instructions.",
    )
    parser.add_argument(
        "--category-descriptions",
        default=None,
        help=(
            "JSON mapping from category name to visual description. Required for "
            "instruction enhancement unless descriptions are embedded in episodes."
        ),
    )
    args = parser.parse_args()
    payload = evaluate_checkpoint(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        episodes_path=args.episodes,
        device=args.device,
        batch_size=args.batch_size,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens,
        attention=args.attention,
        data_root=args.data_root,
        visual_enhancement=args.visual_enhancement,
        instruction_enhancement=args.instruction_enhancement,
        category_descriptions_path=args.category_descriptions,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for shot, summary in payload["metrics_by_shot"].items():
        model_map = summary["coco_map"]["model"]
        ranking_map = summary["coco_map"]["ranking"]
        print(
            f"{shot}-shot: F1@Mean={summary['f1_mean_over_iou']:.4f}, "
            f"mAP={model_map['map_50_95']:.4f}, AP50={model_map['map_50']:.4f}, "
            f"ranking-mAP={ranking_map['map_50_95']:.4f}"
        )
    print(f"Saved evaluation to {output}")


if __name__ == "__main__":
    main()
