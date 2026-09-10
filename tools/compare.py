#!/usr/bin/env python3
"""Compare two result files produced by ``tools/evaluate_coco.py``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    before = json.loads(Path(args.before).read_text(encoding="utf-8"))
    after = json.loads(Path(args.after).read_text(encoding="utf-8"))
    if before.get("protocol") != after.get("protocol"):
        raise ValueError("cannot compare different protocols")
    for key in (
        "prompt_template_version",
        "episodes_sha256",
        "min_pixels",
        "max_pixels",
        "max_new_tokens",
    ):
        if before.get(key) != after.get(key):
            raise ValueError(f"cannot compare results with different {key}")
    comparison = {"before": args.before, "after": args.after, "by_shot": {}}
    for shot in sorted(
        set(before["metrics_by_shot"]) & set(after["metrics_by_shot"]),
        key=int,
    ):
        before_metrics = before["metrics_by_shot"][shot]
        after_metrics = after["metrics_by_shot"][shot]
        comparison["by_shot"][shot] = {
            "before": before_metrics,
            "after": after_metrics,
            "delta": {
                "f1_mean_over_iou": (
                    after_metrics["f1_mean_over_iou"]
                    - before_metrics["f1_mean_over_iou"]
                ),
                "count_accuracy": (
                    after_metrics["count_accuracy"] - before_metrics["count_accuracy"]
                ),
                "mean_matched_iou": (
                    after_metrics["mean_matched_iou"]
                    - before_metrics["mean_matched_iou"]
                ),
            },
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(comparison["by_shot"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
