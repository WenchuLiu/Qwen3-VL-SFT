#!/usr/bin/env python3
"""Evaluate fixed COCO episodes across persistent GPU replicas."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwen3vl_sft.evaluation.coco.generation import load_episodes, result_payload
from qwen3vl_sft.evaluation.coco.metrics import evaluate_episode_predictions
from qwen3vl_sft.evaluation.fewshot.cli import GenerationRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--adapter-path', required=True)
    parser.add_argument('--episodes', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data-root', default=str(ROOT / 'data'))
    parser.add_argument('--num-gpus', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--min-pixels', type=int, default=4096)
    parser.add_argument('--max-pixels', type=int, default=640000)
    parser.add_argument('--max-new-tokens', type=int, default=1024)
    args = parser.parse_args()
    metadata, records = load_episodes(args.episodes, media_root=args.data_root)
    started = time.monotonic()
    with GenerationRunner(
        model_path=args.model_path, adapter_path=args.adapter_path,
        device='cuda:0', num_gpus=args.num_gpus, attention='sdpa',
        batch_size=args.batch_size, min_pixels=args.min_pixels, max_pixels=args.max_pixels,
    ) as runner:
        responses = runner.generate(records, max_new_tokens=args.max_new_tokens)
        device = runner.device_label
    result = evaluate_episode_predictions(
        records, responses, coco_annotations_path=metadata['query_annotations'],
    )
    payload = result_payload(
        result, model_path=args.model_path, adapter_path=args.adapter_path,
        episodes_path=str(Path(args.episodes).resolve()), device=device,
        batch_size=args.batch_size, min_pixels=args.min_pixels, max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens, coco_annotations_path=metadata['query_annotations'],
        runtime_seconds=time.monotonic() - started,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    for shot, metrics in result['metrics_by_shot'].items():
        print(f'{shot}-shot: mAP={metrics["coco_map"]["model"]["map_50_95"]:.6f} '
              f'F1@Mean={metrics["f1_mean_over_iou"]:.6f}', flush=True)


if __name__ == '__main__':
    main()
