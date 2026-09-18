"""Training orchestration for Qwen3-VL SFT.

This module is deliberately limited to runtime wiring.  Argument parsing,
data construction, model policy, and Trainer implementations live in their
respective package modules.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch
from transformers import TrainingArguments, set_seed

from .arguments import build_train_parser, data_config_from_args
from .data import make_data_module
from ..model.loader import (
    configure_trainable_parameters,
    load_model_and_processor,
    save_model_and_processor,
)
from .trainer import GenerationEvalTrainer, TrainingTelemetryTrainer


logger = logging.getLogger(__name__)


def _disable_proxy_environment() -> None:
    """Prevent training and metric logging from inheriting a stale proxy."""
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ):
        os.environ.pop(name, None)


def _report_to(value: str) -> list[str] | str:
    if not value or value.lower() == "none":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _trainer_report_to(value: str) -> list[str]:
    """Return Trainer integrations other than SwanLab.

    SwanLab is logged by ``TrainingTelemetryCallback`` so that its payload can
    be restricted to the small set of metrics used for SFT monitoring.
    """
    return [item for item in _report_to(value) if item.lower() != "swanlab"]


def _has_checkpoint(output_dir: Path) -> bool:
    return any(
        path.is_dir() and (path / "trainer_state.json").is_file()
        for path in output_dir.glob("checkpoint-*")
    )


def _init_swanlab_if_requested(args) -> None:
    """Create the run before the custom telemetry callback starts logging.

    SwanLab 0.10 raises when ``get_run`` is called before initialization.
    Initialize only the world-process-zero rank; metric handling is kept in the
    custom callback so only the requested metrics are uploaded.
    """
    if not any(item.lower() == "swanlab" for item in _report_to(args.report_to)):
        return
    # ``torchrun`` exports RANK before the process group is initialized. Check
    # it first so multi-GPU launches do not create one SwanLab run per worker.
    try:
        launch_rank = int(os.getenv("RANK", "0"))
    except ValueError:
        launch_rank = 0
    if launch_rank != 0:
        return
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_rank() != 0
    ):
        return

    import swanlab

    try:
        active_run = swanlab.get_run()
    except RuntimeError:
        active_run = None
    if active_run is not None:
        return

    # SWANLAB_PROJECT is parsed as a structured setting by newer SDKs. Use the
    # scalar compatibility variable and remove the conflicting name before
    # calling init; the Transformers callback only needs the already-active run.
    project = os.getenv("SWANLAB_PROJ_NAME") or os.getenv("SWANLAB_PROJECT")
    os.environ.pop("SWANLAB_PROJECT", None)
    init_args = {}
    if project:
        init_args["project"] = project
    workspace = os.getenv("SWANLAB_WORKSPACE")
    if workspace:
        init_args["workspace"] = workspace
    if args.run_name:
        init_args["experiment_name"] = args.run_name
    swanlab.init(**init_args)


def _training_args(args):
    eval_strategy = args.eval_strategy
    if args.eval_mode == "none":
        eval_strategy = "no"
    if args.eval_mode == "generation" and not args.coco_eval_episodes:
        raise ValueError("--eval-mode generation requires --coco-eval-episodes")
    if args.eval_mode in {"loss", "generation"} and args.eval_strategy == "no":
        raise ValueError(
            f"--eval-mode {args.eval_mode} requires --eval-strategy epoch or steps"
        )
    if args.eval_mode == "loss" and args.eval_dataset == [] and args.eval_ratio == 0:
        raise ValueError("loss evaluation requires --eval-dataset or --eval-ratio > 0")
    if args.bf16 and args.fp16:
        raise ValueError("--bf16 and --fp16 cannot both be enabled")

    return TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=False,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy=eval_strategy,
        eval_steps=args.eval_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        remove_unused_columns=False,
        label_names=["labels"],
        bf16=args.bf16,
        fp16=args.fp16,
        tf32=args.tf32 and torch.cuda.is_available(),
        seed=args.seed,
        run_name=args.run_name,
        report_to=_trainer_report_to(args.report_to),
    )


def train(args) -> None:
    _disable_proxy_environment()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    if args.bf16 and not torch.cuda.is_available():
        raise RuntimeError("bf16 training requires CUDA; pass --bf16 false for a CPU smoke test")

    model, processor = load_model_and_processor(args)
    model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model = configure_trainable_parameters(model, args)

    data_args = data_config_from_args(args)
    data_module = make_data_module(processor, data_args)
    training_args = _training_args(args)
    _init_swanlab_if_requested(args)
    swanlab_enabled = any(
        item.lower() == "swanlab" for item in _report_to(args.report_to)
    )

    if args.eval_mode == "generation":
        if args.eval_strategy == "no":
            raise ValueError("generation evaluation needs --eval-strategy epoch or steps")
        # Trainer validates that an eval dataset exists when an eval schedule is
        # enabled. The overridden evaluate() never reads this sentinel.
        data_module["eval_dataset"] = [None]
        trainer = GenerationEvalTrainer(
            model=model,
            processing_class=processor.tokenizer,
            args=training_args,
            **data_module,
            generation_eval_episodes=args.coco_eval_episodes,
            generation_eval_processor=processor,
            generation_eval_batch_size=args.coco_eval_batch_size,
            generation_eval_min_pixels=args.coco_eval_min_pixels or args.min_pixels,
            generation_eval_max_pixels=args.coco_eval_max_pixels or args.max_pixels,
            generation_eval_max_new_tokens=args.coco_eval_max_new_tokens,
            generation_eval_model_path=args.model_name_or_path,
            swanlab_enabled=swanlab_enabled,
        )
    else:
        trainer = TrainingTelemetryTrainer(
            model=model,
            processing_class=processor.tokenizer,
            args=training_args,
            **data_module,
            swanlab_enabled=swanlab_enabled,
        )

    if args.resume_from_checkpoint:
        checkpoint = Path(args.resume_from_checkpoint).expanduser().resolve()
        if not (checkpoint / "trainer_state.json").is_file():
            raise FileNotFoundError(f"not a Trainer checkpoint: {checkpoint}")
        resume = str(checkpoint)
    elif args.resume_training and _has_checkpoint(output_dir):
        resume = True
    else:
        resume = None
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_state()
    model.config.use_cache = True
    save_model_and_processor(trainer, processor, str(output_dir))
    (output_dir / "run_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = build_train_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    train(args)


if __name__ == "__main__":
    main()

