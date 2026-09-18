"""Command-line configuration for focused Qwen3-VL SFT runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


DEFAULT_MIN_PIXELS = 28 * 28 * 4
DEFAULT_MAX_PIXELS = 800 * 800


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


@dataclass
class DataConfig:
    dataset: list[str]
    eval_dataset: list[str]
    data_root: str | None
    eval_mode: str
    eval_ratio: float
    eval_seed: int
    model_max_length: int
    min_pixels: int
    max_pixels: int
    video_min_pixels: int
    video_max_pixels: int
    video_min_frames: int
    video_max_frames: int
    video_fps: float


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("data")
    group.add_argument(
        "--dataset", nargs="+", required=True,
        help="One or more JSON/JSONL annotation files in the Qwen VL schema.",
    )
    group.add_argument(
        "--eval-dataset", nargs="*", default=[],
        help="Optional held-out JSON/JSONL files for loss evaluation.",
    )
    group.add_argument("--data-root", default=None)
    group.add_argument(
        "--eval-mode", choices=("none", "loss", "generation"), default="none",
        help="none: no eval; loss: teacher-forced LM loss; generation: fixed episode F1 and mAP.",
    )
    group.add_argument("--eval-ratio", type=float, default=0.0)
    group.add_argument("--eval-seed", type=int, default=42)
    group.add_argument("--model-max-length", type=int, default=8192)
    group.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    group.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    group.add_argument("--video-min-pixels", type=int, default=256 * 28 * 28)
    group.add_argument("--video-max-pixels", type=int, default=1024 * 28 * 28)
    group.add_argument("--video-min-frames", type=int, default=4)
    group.add_argument("--video-max-frames", type=int, default=8)
    group.add_argument("--video-fps", type=float, default=2.0)


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("training")
    group.add_argument("--model-name-or-path", required=True)
    group.add_argument("--output-dir", required=True)
    group.add_argument("--cache-dir", default=None)
    group.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )
    group.add_argument("--num-train-epochs", type=float, default=12.0)
    group.add_argument("--max-steps", type=int, default=-1)
    group.add_argument("--per-device-train-batch-size", type=int, default=1)
    group.add_argument("--per-device-eval-batch-size", type=int, default=1)
    group.add_argument("--gradient-accumulation-steps", type=int, default=1)
    group.add_argument("--learning-rate", type=float, default=2e-4)
    group.add_argument("--weight-decay", type=float, default=0.0)
    group.add_argument("--warmup-ratio", type=float, default=0.03)
    group.add_argument("--lr-scheduler-type", default="cosine")
    group.add_argument("--optim", default="adamw_torch")
    group.add_argument("--max-grad-norm", type=float, default=1.0)
    group.add_argument("--logging-steps", type=int, default=10)
    group.add_argument("--save-strategy", choices=("steps", "epoch", "no"), default="epoch")
    group.add_argument("--save-steps", type=int, default=500)
    group.add_argument("--save-total-limit", type=int, default=2)
    group.add_argument("--eval-strategy", choices=("no", "steps", "epoch"), default="no")
    group.add_argument("--eval-steps", type=int, default=None)
    group.add_argument("--dataloader-num-workers", type=int, default=2)
    group.add_argument("--gradient-checkpointing", type=str_to_bool, default=True)
    group.add_argument("--ddp-find-unused-parameters", type=str_to_bool, default=False)
    group.add_argument("--resume-from-checkpoint", default=None)
    group.add_argument("--resume-training", type=str_to_bool, default=True)
    group.add_argument("--seed", type=int, default=42)
    group.add_argument("--run-name", default=None)
    group.add_argument("--report-to", default="swanlab")
    group.add_argument("--bf16", type=str_to_bool, default=True)
    group.add_argument("--fp16", type=str_to_bool, default=False)
    group.add_argument("--tf32", type=str_to_bool, default=True)

    lora = parser.add_argument_group("lora and trainable modules")
    lora.add_argument("--lora-enable", type=str_to_bool, default=True)
    lora.add_argument("--lora-r", type=int, default=64)
    lora.add_argument("--lora-alpha", type=int, default=128)
    lora.add_argument("--lora-dropout", type=float, default=0.05)
    lora.add_argument("--tune-mm-vision", type=str_to_bool, default=False)
    lora.add_argument("--tune-mm-mlp", type=str_to_bool, default=False)
    lora.add_argument("--tune-mm-llm", type=str_to_bool, default=True)


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Qwen3-VL with the official multimodal conversation schema."
    )
    add_data_arguments(parser)
    add_training_arguments(parser)
    parser.add_argument("--coco-eval-episodes", default=None)
    parser.add_argument("--coco-eval-batch-size", type=int, default=1)
    parser.add_argument("--coco-eval-min-pixels", type=int, default=None)
    parser.add_argument("--coco-eval-max-pixels", type=int, default=None)
    parser.add_argument("--coco-eval-max-new-tokens", type=int, default=1024)
    return parser


def data_config_from_args(args: argparse.Namespace) -> DataConfig:
    return DataConfig(
        dataset=args.dataset,
        eval_dataset=args.eval_dataset,
        data_root=args.data_root,
        eval_mode=args.eval_mode,
        eval_ratio=args.eval_ratio,
        eval_seed=args.eval_seed,
        model_max_length=args.model_max_length,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        video_min_pixels=args.video_min_pixels,
        video_max_pixels=args.video_max_pixels,
        video_min_frames=args.video_min_frames,
        video_max_frames=args.video_max_frames,
        video_fps=args.video_fps,
    )
