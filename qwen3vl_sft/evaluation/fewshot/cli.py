#!/usr/bin/env python3
"""Few-shot evaluation application for local cross-domain datasets.

Each dataset's ``annotations/{1,2,4}_shot.json`` file is used as the support
pool and ``annotations/test.json`` is used as the query set.  Evaluation builds
one category-conditioned episode for every query image/category pair, writes a
reusable manifest, and stores the generation result under a per-dataset,
per-shot work directory.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import queue
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen3vl_sft.config import DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS
from qwen3vl_sft.evaluation.coco.data import (
    build_fixed_support_eval_records,
    load_coco_frames,
    parse_shots,
    write_json,
)
from qwen3vl_sft.evaluation.coco.generation import file_sha256
from qwen3vl_sft.evaluation.coco.metrics import evaluate_episode_predictions
from qwen3vl_sft.evaluation.coco.protocol import (
    DETPO_PROMPT_VERSION,
    PROMPT_TEMPLATE_VERSION,
    PROTOCOL_NAME,
    load_category_descriptions,
)

from .config import (
    DEFAULT_DATASETS,
    _dataset_dir,
    _dataset_image_roots,
    _max_new_tokens_for_dataset,
    _result_is_complete,
)


def _load_model(model_path: str, device: str, attention: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=device,
        attn_implementation=attention,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor


def _generation_worker_loop(
    rank: int,
    device: str,
    model_path: str,
    attention: str,
    task_queue,
    result_queue,
    batch_size: int,
    min_pixels: int,
    max_pixels: int,
    visual_enhancement: bool,
    instruction_enhancement: bool = False,
    detpo: bool = False,
    category_descriptions: dict[str, str] | None = None,
) -> None:
    """Load one model replica and process generation tasks until shutdown."""
    try:
        from qwen3vl_sft.evaluation.coco.generation import generate_responses

        print(f"[GPU {device}] loading model replica", flush=True)
        model, processor = _load_model(model_path, device, attention)
        print(f"[GPU {device}] model ready", flush=True)

        while True:
            item = task_queue.get()
            if item is None:
                return
            task_id, records, max_new_tokens = item

            def report_progress(done: int, total: int) -> None:
                if done == total or done % 100 == 0:
                    print(
                        f"[GPU {device}] task={task_id} {done}/{total}",
                        flush=True,
                    )

            responses = generate_responses(
                records,
                model,
                processor,
                device,
                batch_size=batch_size,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                max_new_tokens=max_new_tokens,
                visual_enhancement=visual_enhancement,
                instruction_enhancement=instruction_enhancement,
                detpo=detpo,
                category_descriptions=category_descriptions,
                progress_callback=report_progress,
            )
            result_queue.put((rank, task_id, responses, None))
    except BaseException:
        result_queue.put((rank, None, None, traceback.format_exc()))


class GenerationRunner:
    """Run generation on one model or on persistent replicas across GPUs."""

    def __init__(
        self,
        *,
        model_path: str,
        device: str,
        num_gpus: int,
        attention: str,
        batch_size: int,
        min_pixels: int,
        max_pixels: int,
        visual_enhancement: bool = False,
        instruction_enhancement: bool = False,
        detpo: bool = False,
        category_descriptions: dict[str, str] | None = None,
    ) -> None:
        import torch

        if num_gpus < 1:
            raise ValueError("num_gpus must be positive")
        self.model_path = model_path
        self.attention = attention
        self.batch_size = batch_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.visual_enhancement = visual_enhancement
        self.instruction_enhancement = instruction_enhancement
        self.detpo = detpo
        self.category_descriptions = category_descriptions
        self.task_id = 0
        self.parallel = num_gpus > 1
        self.model = None
        self.processor = None
        self.processes = []
        self.task_queues = []
        self.result_queue = None

        if not self.parallel:
            print(f"[GPU {device}] loading model replica", flush=True)
            self.model, self.processor = _load_model(model_path, device, attention)
            print(f"[GPU {device}] model ready", flush=True)
            self.devices = [device]
            return

        if not torch.cuda.is_available():
            raise RuntimeError("--num-gpus > 1 requires CUDA")
        available = torch.cuda.device_count()
        if available < num_gpus:
            raise RuntimeError(
                f"requested {num_gpus} GPUs, but only {available} are visible"
            )
        self.devices = [f"cuda:{index}" for index in range(num_gpus)]
        context = mp.get_context("spawn")
        self.result_queue = context.Queue()
        for rank, worker_device in enumerate(self.devices):
            task_queue = context.Queue()
            process = context.Process(
                target=_generation_worker_loop,
                args=(
                    rank,
                    worker_device,
                    model_path,
                    attention,
                    task_queue,
                    self.result_queue,
                    batch_size,
                    min_pixels,
                    max_pixels,
                    visual_enhancement,
                    instruction_enhancement,
                    detpo,
                    category_descriptions,
                ),
            )
            process.start()
            self.task_queues.append(task_queue)
            self.processes.append(process)
        print(f"Started {num_gpus} persistent GPU workers: {self.devices}", flush=True)

    @property
    def device_label(self) -> str:
        return ",".join(self.devices)

    def generate(self, records: list[dict], *, max_new_tokens: int) -> list[str]:
        from qwen3vl_sft.evaluation.coco.generation import generate_responses

        if not self.parallel:
            return generate_responses(
                records,
                self.model,
                self.processor,
                self.devices[0],
                batch_size=self.batch_size,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                max_new_tokens=max_new_tokens,
                visual_enhancement=self.visual_enhancement,
                instruction_enhancement=self.instruction_enhancement,
                detpo=self.detpo,
                category_descriptions=self.category_descriptions,
            )

        task_id = self.task_id
        self.task_id += 1
        chunks = [records[rank::len(self.processes)] for rank in range(len(self.processes))]
        for task_queue, chunk in zip(self.task_queues, chunks):
            task_queue.put((task_id, chunk, max_new_tokens))

        responses_by_rank: dict[int, list[str]] = {}
        while len(responses_by_rank) < len(self.processes):
            try:
                rank, returned_task_id, responses, error = self.result_queue.get(timeout=5)
            except queue.Empty:
                if any(not process.is_alive() for process in self.processes):
                    self.close(terminate=True)
                    raise RuntimeError("a GPU worker exited without returning results")
                continue
            if error is not None:
                self.close(terminate=True)
                raise RuntimeError(f"GPU worker {rank} failed:\n{error}")
            if returned_task_id != task_id:
                self.close(terminate=True)
                raise RuntimeError(
                    f"worker {rank} returned task {returned_task_id}, expected {task_id}"
                )
            responses_by_rank[rank] = responses

        combined: list[str] = [""] * len(records)
        for rank, responses in responses_by_rank.items():
            for index, response in zip(range(rank, len(records), len(self.processes)), responses):
                combined[index] = response
        return combined

    def close(self, *, terminate: bool = False) -> None:
        if not self.parallel:
            self.model = None
            self.processor = None
            return
        if not terminate:
            for task_queue in self.task_queues:
                task_queue.put(None)
        else:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
        for process in self.processes:
            process.join(timeout=30)
        for process in self.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for task_queue in self.task_queues:
            task_queue.close()
        if self.result_queue is not None:
            self.result_queue.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback_value):
        self.close(terminate=exc_type is not None)


def _lookup_description(category: str, descriptions: dict[str, str]) -> str | None:
    """Look up a category description while tolerating case-only name drift."""
    description = descriptions.get(category)
    if description is not None:
        return description
    folded_category = category.casefold()
    for key, value in descriptions.items():
        if key.casefold() == folded_category:
            return value
    return None


def _resolve_detpo_prompt_path(
    prompt_root: Path,
    dataset_name: str,
    dataset_dir: Path,
    shot: int,
) -> Path:
    """Resolve the DetPO prompt file for one dataset/shot episode."""
    if shot < 1:
        raise ValueError("DetPO prompts are available only for 1/2/4-shot evaluation")
    shot_dir = prompt_root / f"{shot}-shot"
    stems = []
    for stem in (dataset_dir.name, dataset_name):
        if stem and stem.casefold() not in {value.casefold() for value in stems}:
            stems.append(stem)

    for stem in stems:
        candidate = shot_dir / f"all_refined_class_instructions_{stem}.json"
        if candidate.is_file():
            return candidate.resolve()

    # Keep the resolver useful when a dataset directory uses a different case
    # from the generated file name.
    expected_names = {
        f"all_refined_class_instructions_{stem}".casefold() for stem in stems
    }
    for candidate in sorted(shot_dir.glob("all_refined_class_instructions_*.json")):
        if candidate.stem.casefold() in expected_names:
            return candidate.resolve()

    expected = ", ".join(
        f"{shot_dir / f'all_refined_class_instructions_{stem}.json'}" for stem in stems
    )
    raise FileNotFoundError(
        f"DetPO prompt file not found for dataset {dataset_name!r}, {shot}-shot; "
        f"expected one of: {expected}"
    )


def _embed_detpo_descriptions(
    records: list[dict], descriptions: dict[str, str], dataset_name: str
) -> list[dict]:
    """Attach the selected DetPO description to every episode query category."""
    missing = sorted(
        {
            str(record.get("category"))
            for record in records
            if isinstance(record.get("category"), str)
            and _lookup_description(record["category"], descriptions) is None
        }
    )
    if missing:
        raise ValueError(
            f"DetPO prompt file does not define categories for {dataset_name!r}: "
            + ", ".join(missing)
        )

    enriched = []
    for record in records:
        category = record["category"]
        description = _lookup_description(category, descriptions)
        # The missing-category check above guarantees this is present. Keep the
        # assertion local so the manifest never contains a null description.
        assert description is not None
        enriched.append({**record, "category_description": description})
    return enriched


def _build_manifest(
    *,
    dataset_name: str,
    dataset_dir: Path,
    shot: int,
    output: Path,
    seed: int,
    min_box_area_ratio: float,
    max_query_pairs: int | None = None,
    detpo_prompt_path: Path | None = None,
    detpo_prompt_sha256: str | None = None,
    detpo_descriptions: dict[str, str] | None = None,
) -> None:
    annotations_dir = dataset_dir / "annotations"
    support_annotation = annotations_dir / (
        "1_shot.json" if shot == 0 else f"{shot}_shot.json"
    )
    query_annotation = annotations_dir / "test.json"
    support_image_root, query_image_root = _dataset_image_roots(dataset_dir)
    for path in (
        support_annotation,
        query_annotation,
        support_image_root,
        query_image_root,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    support_frames = load_coco_frames(
        support_annotation,
        support_image_root,
        min_box_area_ratio=min_box_area_ratio,
    )
    query_frames = load_coco_frames(
        query_annotation,
        query_image_root,
        min_box_area_ratio=min_box_area_ratio,
    )
    records = build_fixed_support_eval_records(
        support_frames,
        query_frames,
        shot=shot,
    )
    if detpo_descriptions is not None:
        records = _embed_detpo_descriptions(records, detpo_descriptions, dataset_name)
    if max_query_pairs is not None:
        records = records[:max_query_pairs]
    is_detpo = detpo_descriptions is not None
    write_json(
        output,
        {
            "protocol": PROTOCOL_NAME,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "dataset": dataset_name,
            "support_annotations": str(support_annotation.resolve()),
            "query_annotations": str(query_annotation.resolve()),
            "support_image_root": str(support_image_root.resolve()),
            "query_image_root": str(query_image_root.resolve()),
            "shots": [shot],
            "seed": seed,
            "min_box_area_ratio": min_box_area_ratio,
            "all_query_pairs": True,
            "max_query_pairs": max_query_pairs,
            "detpo": is_detpo,
            "prompt_method": "DetPO" if is_detpo else "baseline",
            "detpo_prompt_path": (
                str(detpo_prompt_path.resolve()) if detpo_prompt_path is not None else None
            ),
            "detpo_prompt_sha256": detpo_prompt_sha256,
            "detpo_prompt_version": DETPO_PROMPT_VERSION if is_detpo else None,
            "records": records,
        },
    )
    method = "DetPO " if is_detpo else ""
    print(
        f"[{dataset_name} {shot}-shot] {method}built {len(records)} episodes: {output}",
        flush=True,
    )


def _evaluate_one(
    *,
    runner: GenerationRunner,
    model_path: str,
    dataset_name: str,
    manifest: Path,
    result_path: Path,
    batch_size: int,
    min_pixels: int,
    max_pixels: int,
    max_new_tokens: int,
    attention: str,
    visual_enhancement: bool,
    instruction_enhancement: bool = False,
    detpo: bool = False,
    category_descriptions_path: str | None = None,
    category_descriptions_sha256: str | None = None,
    detpo_prompt_path: str | None = None,
    detpo_prompt_sha256: str | None = None,
    detpo_prompt_version: str | None = None,
) -> dict:
    from qwen3vl_sft.evaluation.coco.generation import load_episodes, result_payload

    metadata, records = load_episodes(manifest)
    started = time.monotonic()
    responses = runner.generate(records, max_new_tokens=max_new_tokens)
    result = evaluate_episode_predictions(
        records,
        responses,
        coco_annotations_path=metadata.get("query_annotations"),
    )
    payload = result_payload(
        result,
        model_path=model_path,
        adapter_path=None,
        episodes_path=str(manifest.resolve()),
        device=runner.device_label,
        batch_size=batch_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        max_new_tokens=max_new_tokens,
        visual_enhancement=visual_enhancement,
        instruction_enhancement=instruction_enhancement,
        detpo=detpo,
        category_descriptions_path=category_descriptions_path,
        category_descriptions_sha256=category_descriptions_sha256,
        detpo_prompt_path=detpo_prompt_path,
        detpo_prompt_sha256=detpo_prompt_sha256,
        detpo_prompt_version=detpo_prompt_version,
        coco_annotations_path=metadata.get("query_annotations"),
        runtime_seconds=time.monotonic() - started,
    )
    payload.update(
        {
            "dataset": dataset_name,
            "attention": attention,
            "support_annotations": metadata.get("support_annotations"),
            "query_annotations": metadata.get("query_annotations"),
        }
    )
    write_json(result_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=ROOT / "outputs" / "eval" / "fewshot" / "qwen3-vl-4b-base-fewshot",
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument(
        "--shots",
        nargs="+",
        default=None,
        help="Shot counts; defaults to 0/1/2/4, or 1/2/4 for --detpo.",
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of persistent model replicas; replicas use cuda:0..N-1.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help=(
            "Global generation-budget override. If omitted, use the "
            "Qwen3-VL dataset defaults: 1024 for ArTaxOr/Clipart1k/FISH/"
            "NEU-DET/UODD and 2048 for DIOR/VISUALDIOR."
        ),
    )
    parser.add_argument(
        "--flat-work-dir",
        action="store_true",
        help="For one dataset, write shot results directly under --work-dir.",
    )
    parser.add_argument(
        "--min-box-area-ratio",
        type=float,
        default=0.0,
        help="Keep all annotated boxes by default; set >0 to filter tiny boxes.",
    )
    parser.add_argument(
        "--attention",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )
    parser.add_argument(
        "--max-query-pairs",
        type=int,
        default=None,
        help="Optional positive query-pair limit for smoke tests.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a dataset/shot when its result.json is already complete.",
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
        help="Add the requested category description to the final query instruction.",
    )
    parser.add_argument(
        "--category-descriptions",
        type=Path,
        default=None,
        help=(
            "JSON mapping from category name to visual description; required with "
            "--instruction-enhancement for local few-shot datasets."
        ),
    )
    parser.add_argument(
        "--detpo",
        action="store_true",
        help=(
            "Use the per-dataset, per-shot DetPO prompt files from "
            "--detpo-prompts."
        ),
    )
    parser.add_argument(
        "--detpo-prompts",
        "--detpo-prompt-root",
        type=Path,
        default=ROOT / "docs" / "cross-domain-instructions",
        help=(
            "Root containing {1,2,4}-shot/all_refined_class_instructions_*.json "
            "files used by --detpo."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or (
        args.max_new_tokens is not None and args.max_new_tokens < 1
    ):
        raise ValueError("batch size and max new tokens must be positive")
    if args.num_gpus < 1:
        raise ValueError("num gpus must be positive")
    if args.min_box_area_ratio < 0 or args.min_box_area_ratio >= 1:
        raise ValueError("min box area ratio must be in [0, 1)")
    if args.max_query_pairs is not None and args.max_query_pairs < 1:
        raise ValueError("max query pairs must be positive")
    shot_values = args.shots
    if shot_values is None:
        shot_values = ["1", "2", "4"] if args.detpo else ["0", "1", "2", "4"]
    shots = parse_shots([str(value) for value in shot_values], allow_zero=True)
    if args.visual_enhancement and 0 in shots:
        raise ValueError("--ve/--visual-enhancement requires at least one support shot")
    if args.detpo and args.instruction_enhancement:
        raise ValueError("--detpo and --instruction-enhancement are mutually exclusive")
    if args.detpo and args.visual_enhancement:
        raise ValueError(
            "--detpo uses the original single-image prompt and cannot be combined "
            "with --ve/--visual-enhancement"
        )
    if args.detpo and args.category_descriptions is not None:
        raise ValueError(
            "--detpo selects per-shot prompt files; do not combine it with "
            "--category-descriptions"
        )
    if args.detpo and 0 in shots:
        raise ValueError("--detpo requires shots 1, 2, or 4; no 0-shot prompt is provided")
    if args.instruction_enhancement and args.category_descriptions is None:
        raise ValueError(
            "--ie/--instruction-enhancement requires --category-descriptions "
            "for local few-shot evaluation"
        )
    category_descriptions_path = None
    category_descriptions = None
    category_descriptions_sha256 = None
    if args.category_descriptions is not None:
        category_descriptions_path = args.category_descriptions.expanduser().resolve()
        category_descriptions = load_category_descriptions(category_descriptions_path)
        category_descriptions_sha256 = file_sha256(category_descriptions_path)
    detpo_prompt_root = args.detpo_prompts.expanduser().resolve()
    if args.detpo and not detpo_prompt_root.is_dir():
        raise NotADirectoryError(f"DetPO prompt root not found: {detpo_prompt_root}")
    work_dir = args.work_dir.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    model_path = (
        str(Path(args.model_path).expanduser().resolve())
        if Path(args.model_path).exists()
        else args.model_path
    )

    datasets = [_dataset_dir(data_root, name) for name in args.datasets]
    flat_work_dir = args.flat_work_dir and len(datasets) == 1
    max_new_tokens_by_dataset = {
        dataset_name: _max_new_tokens_for_dataset(
            dataset_name,
            args.max_new_tokens,
        )
        for dataset_name, _ in datasets
    }
    tasks = []
    for dataset_name, dataset_dir in datasets:
        for shot in shots:
            shot_dir = (
                work_dir / f"{shot}shot"
                if flat_work_dir
                else work_dir / dataset_name / f"{shot}shot"
            )
            task = {
                "dataset": dataset_name,
                "dataset_dir": dataset_dir,
                "shot": shot,
                "manifest": shot_dir / "episodes.json",
                "result": shot_dir / "result.json",
                "max_new_tokens": max_new_tokens_by_dataset[dataset_name],
            }
            if args.detpo:
                prompt_path = _resolve_detpo_prompt_path(
                    detpo_prompt_root,
                    dataset_name,
                    dataset_dir,
                    shot,
                )
                task["detpo_prompt_path"] = prompt_path
                task["detpo_prompt_sha256"] = file_sha256(prompt_path)
                task["detpo_descriptions"] = load_category_descriptions(prompt_path)
            tasks.append(task)

    write_json(
        work_dir / "config.json",
        {
            "model_path": model_path,
            "data_root": str(data_root),
            "work_dir": str(work_dir),
            "datasets": [name for name, _ in datasets],
            "shots": shots,
            "seed": args.seed,
            "device": args.device,
            "num_gpus": args.num_gpus,
            "batch_size": args.batch_size,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "max_new_tokens": args.max_new_tokens,
            "max_new_tokens_by_dataset": max_new_tokens_by_dataset,
            "flat_work_dir": flat_work_dir,
            "min_box_area_ratio": args.min_box_area_ratio,
            "attention": args.attention,
            "all_query_pairs": True,
            "max_query_pairs": args.max_query_pairs,
            "visual_enhancement": args.visual_enhancement,
            "instruction_enhancement": args.instruction_enhancement,
            "detpo": args.detpo,
            "detpo_prompt_version": DETPO_PROMPT_VERSION if args.detpo else None,
            "prompt_method": "DetPO" if args.detpo else (
                "IE" if args.instruction_enhancement else "baseline"
            ),
            "detpo_prompts_root": str(detpo_prompt_root) if args.detpo else None,
            "category_descriptions_path": (
                str(category_descriptions_path) if category_descriptions_path else None
            ),
            "category_descriptions_sha256": category_descriptions_sha256,
        },
    )

    pending = [
        task
        for task in tasks
        if not args.skip_existing
        or not _result_is_complete(
            task["result"],
            task["max_new_tokens"],
            expected_visual_enhancement=args.visual_enhancement,
            expected_instruction_enhancement=args.instruction_enhancement,
            expected_detpo=args.detpo,
            expected_category_descriptions_sha256=(
                category_descriptions_sha256 if args.instruction_enhancement else None
            ),
            expected_detpo_prompt_sha256=(
                task.get("detpo_prompt_sha256") if args.detpo else None
            ),
            expected_detpo_prompt_version=(
                DETPO_PROMPT_VERSION if args.detpo else None
            ),
        )
    ]
    if not pending:
        print(f"All requested results already exist under {work_dir}")
        return

    summary_rows = []
    summary_path = work_dir / "summary.json"
    print(
        f"Starting evaluation with {args.num_gpus} GPU replica(s)",
        flush=True,
    )
    with GenerationRunner(
        model_path=model_path,
        device=args.device,
        num_gpus=args.num_gpus,
        attention=args.attention,
        batch_size=args.batch_size,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        visual_enhancement=args.visual_enhancement,
        instruction_enhancement=args.instruction_enhancement,
        detpo=args.detpo,
        category_descriptions=category_descriptions,
    ) as runner:
        for task in pending:
            _build_manifest(
                dataset_name=task["dataset"],
                dataset_dir=task["dataset_dir"],
                shot=task["shot"],
                output=task["manifest"],
                seed=args.seed,
                min_box_area_ratio=args.min_box_area_ratio,
                max_query_pairs=args.max_query_pairs,
                detpo_prompt_path=task.get("detpo_prompt_path"),
                detpo_prompt_sha256=task.get("detpo_prompt_sha256"),
                detpo_descriptions=task.get("detpo_descriptions"),
            )
            payload = _evaluate_one(
                runner=runner,
                model_path=model_path,
                dataset_name=task["dataset"],
                manifest=task["manifest"],
                result_path=task["result"],
                batch_size=args.batch_size,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
                max_new_tokens=task["max_new_tokens"],
                attention=args.attention,
                visual_enhancement=args.visual_enhancement,
                instruction_enhancement=args.instruction_enhancement,
                detpo=args.detpo,
                category_descriptions_path=(
                    str(category_descriptions_path) if category_descriptions_path else None
                ),
                category_descriptions_sha256=category_descriptions_sha256,
                detpo_prompt_path=(
                    str(task["detpo_prompt_path"])
                    if args.detpo
                    else None
                ),
                detpo_prompt_sha256=(
                    task["detpo_prompt_sha256"] if args.detpo else None
                ),
                detpo_prompt_version=(
                    DETPO_PROMPT_VERSION if args.detpo else None
                ),
            )
            shot_key = str(task["shot"])
            metrics = payload["metrics_by_shot"][shot_key]
            model_map = metrics["coco_map"]["model"]
            ranking_map = metrics["coco_map"]["ranking"]
            row = {
                "dataset": task["dataset"],
                "shot": task["shot"],
                "ve": args.visual_enhancement,
                "ie": args.instruction_enhancement,
                "detpo": args.detpo,
                "prompt_method": "DetPO" if args.detpo else (
                    "IE" if args.instruction_enhancement else "baseline"
                ),
                "instruction_enhancement": args.instruction_enhancement,
                "episodes": metrics["episodes"],
                "map_50_95": model_map["map_50_95"],
                "map_50": model_map["map_50"],
                "map_75": model_map["map_75"],
                "ranking_map_50_95": ranking_map["map_50_95"],
                "result": str(task["result"].resolve()),
            }
            summary_rows.append(row)
            write_json(summary_path, summary_rows)
            print(
                f"[{task['dataset']} {task['shot']}-shot] "
                f"episodes={row['episodes']} "
                f"mAP={row['map_50_95']:.4f} "
                f"AP50={row['map_50']:.4f} "
                f"AP75={row['map_75']:.4f} "
                f"ranking-mAP={row['ranking_map_50_95']:.4f}",
                flush=True,
            )

    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
