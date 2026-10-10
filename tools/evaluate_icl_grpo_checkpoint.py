#!/usr/bin/env python3
"""Generate fixed-manifest COCO and cross-domain results for one GRPO checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen3vl_sft.evaluation.coco.generation import load_episodes, result_payload
from qwen3vl_sft.evaluation.coco.metrics import evaluate_episode_predictions
from qwen3vl_sft.evaluation.fewshot.cli import GenerationRunner, _evaluate_one


DATASETS = ("ArTaxOr", "Clipart1k", "FISH", "NEU-DET", "UODD")
SHOTS = (0, 1, 2, 4)
COCO_MANIFESTS = {
    shot: f"val_episodes_500_{shot}shot_inst-v5.json" for shot in SHOTS
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def metric_row(*, dataset: str, shot: int, payload: dict, result_path: Path) -> dict:
    metrics = payload["metrics_by_shot"].get(str(shot))
    if not isinstance(metrics, dict):
        raise ValueError(
            f"{dataset} result has no metrics for {shot}-shot: {result_path}"
        )
    return {
        "dataset": dataset,
        "shot": shot,
        "episodes": metrics["episodes"],
        "AP_50_95": metrics["coco_map"]["model"]["map_50_95"],
        "AP_50": metrics["coco_map"]["model"]["map_50"],
        "mean_F1": metrics["f1_mean_over_iou"],
        "result": str(result_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--coco-manifest-dir", type=Path, required=True)
    parser.add_argument("--cross-domain-manifest-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--min-pixels", type=int, default=4096)
    parser.add_argument("--max-pixels", type=int, default=640000)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    model_path = args.model_path.expanduser().resolve()
    adapter_path = args.adapter_path.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    coco_root = args.coco_manifest_dir.expanduser().resolve()
    cross_root = args.cross_domain_manifest_root.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    if not model_path.is_dir():
        raise NotADirectoryError(f"base model directory not found: {model_path}")
    if not adapter_path.is_dir() or not (adapter_path / "adapter_config.json").is_file():
        raise NotADirectoryError(f"LoRA checkpoint not found: {adapter_path}")
    if args.num_gpus < 1 or args.batch_size < 1:
        raise ValueError("num-gpus and batch-size must be positive")
    if args.min_pixels < 1 or args.max_pixels < args.min_pixels:
        raise ValueError("invalid image pixel budget")
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be positive")

    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    started = time.monotonic()
    print(
        f"Evaluating {adapter_path.name} on COCO and {len(DATASETS)} fixed cross-domain "
        f"datasets, shots={list(SHOTS)}",
        flush=True,
    )
    with GenerationRunner(
        model_path=str(model_path),
        adapter_path=str(adapter_path),
        device="cuda:0",
        num_gpus=args.num_gpus,
        attention="sdpa",
        batch_size=args.batch_size,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    ) as runner:
        for shot in SHOTS:
            manifest = coco_root / COCO_MANIFESTS[shot]
            if not manifest.is_file():
                raise FileNotFoundError(f"fixed COCO manifest is missing: {manifest}")
            metadata, records = load_episodes(manifest, media_root=data_root)
            started_one = time.monotonic()
            responses = runner.generate(records, max_new_tokens=args.max_new_tokens)
            metrics = evaluate_episode_predictions(
                records,
                responses,
                coco_annotations_path=metadata.get("query_annotations"),
            )
            path = output_root / "COCO" / f"{shot}shot" / "result.json"
            payload = result_payload(
                metrics,
                model_path=str(model_path),
                adapter_path=str(adapter_path),
                episodes_path=str(manifest),
                device=runner.device_label,
                batch_size=args.batch_size,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
                max_new_tokens=args.max_new_tokens,
                coco_annotations_path=metadata.get("query_annotations"),
                runtime_seconds=time.monotonic() - started_one,
            )
            payload.update(
                {
                    "dataset": "COCO",
                    "shot": shot,
                    "prompt_template_version": "inst-v5",
                    "visual_enhancement": False,
                    "instruction_enhancement": False,
                    "detpo": False,
                    "attention": "sdpa",
                }
            )
            write_json(path, payload)
            rows.append(metric_row(dataset="COCO", shot=shot, payload=payload, result_path=path))
            print(f"[COCO {shot}-shot] saved {path}", flush=True)

        for dataset in DATASETS:
            for shot in SHOTS:
                manifest = cross_root / dataset / f"{shot}shot" / "episodes.json"
                if not manifest.is_file():
                    raise FileNotFoundError(f"fixed cross-domain manifest is missing: {manifest}")
                path = output_root / dataset / f"{shot}shot" / "result.json"
                payload = _evaluate_one(
                    runner=runner,
                    model_path=str(model_path),
                    adapter_path=str(adapter_path),
                    dataset_name=dataset,
                    manifest=manifest,
                    result_path=path,
                    batch_size=args.batch_size,
                    min_pixels=args.min_pixels,
                    max_pixels=args.max_pixels,
                    max_new_tokens=args.max_new_tokens,
                    attention="sdpa",
                    visual_enhancement=False,
                    instruction_enhancement=False,
                    detpo=False,
                )
                rows.append(
                    metric_row(dataset=dataset, shot=shot, payload=payload, result_path=path)
                )
                print(f"[{dataset} {shot}-shot] saved {path}", flush=True)

    elapsed = time.monotonic() - started
    summary = {
        "checkpoint": str(adapter_path),
        "model_path": str(model_path),
        "datasets": ["COCO", *DATASETS],
        "shots": list(SHOTS),
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "max_new_tokens": args.max_new_tokens,
        "deterministic_decoding": True,
        "visual_enhancement": False,
        "instruction_enhancement": False,
        "detpo": False,
        "runtime_seconds": elapsed,
        "results": rows,
    }
    write_json(output_root / "summary.json", summary)
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Completed generation evaluation in {elapsed:.1f}s: {output_root}", flush=True)


if __name__ == "__main__":
    main()
