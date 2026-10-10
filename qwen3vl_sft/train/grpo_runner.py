"""Runtime wiring for score-aware Qwen3-VL GRPO training."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch
from transformers import TrainingArguments, set_seed
from transformers.trainer_callback import TrainerCallback

from .arguments import (
    build_grpo_parser,
    data_config_from_args,
)
from .grpo_data import GRPOPromptDataset
from .grpo_trainer import MultimodalGRPOTrainer
from .rewards import REWARD_FUNCS_REGISTRY
from ..model.loader import (
    configure_trainable_parameters,
    load_model_and_processor,
    save_model_and_processor,
)
from .runner import (
    _disable_proxy_environment,
    _has_checkpoint,
    _init_swanlab_if_requested,
    _report_to,
    _trainer_report_to,
)


logger = logging.getLogger(__name__)


class StopAfterStepCallback(TrainerCallback):
    """Finish a training segment without changing the full-run LR schedule."""

    def __init__(self, step: int):
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.step:
            control.should_training_stop = True
            control.should_save = True
        return control


def _validate_distributed_launch() -> None:
    """Validate DDP topology; every rank loads a complete model replica."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        logger.info(
            "GRPO is running with one process; set NPROC_PER_NODE=4 to use DDP "
            "on the four-GPU 3090server"
        )
        return
    if not torch.cuda.is_available():
        raise ValueError("multi-process GRPO DDP requires CUDA GPUs")
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    device_count = torch.cuda.device_count()
    if local_world_size > device_count:
        raise ValueError(
            f"DDP launch requests {local_world_size} local processes, but only "
            f"{device_count} CUDA devices are visible"
        )
    logger.info(
        "GRPO DDP launch: world_size=%d, local_world_size=%d; "
        "each rank keeps a full model replica",
        world_size,
        local_world_size,
    )


def _resolve_rollout_batch_size(args, world_size: int) -> int:
    """Resolve Rex-Omni-style global rollout prompts into the Trainer microbatch."""
    if world_size < 1:
        raise ValueError("distributed world_size must be positive")

    if args.rollout_batch_size is None:
        per_device_batch_size = int(args.per_device_train_batch_size)
        if per_device_batch_size < 1:
            raise ValueError("per_device_train_batch_size must be positive")
        rollout_batch_size = per_device_batch_size * world_size
    else:
        rollout_batch_size = int(args.rollout_batch_size)
        if rollout_batch_size < 1:
            raise ValueError("rollout_batch_size must be positive")
        if rollout_batch_size % world_size:
            raise ValueError(
                "rollout_batch_size must be divisible by the DDP world_size "
                f"({world_size}); got {rollout_batch_size}"
            )
        per_device_batch_size = rollout_batch_size // world_size
        if per_device_batch_size < 1:
            raise ValueError("rollout_batch_size must provide at least one prompt per rank")
        args.per_device_train_batch_size = per_device_batch_size

    args.rollout_batch_size = rollout_batch_size
    return rollout_batch_size


def _grpo_training_args(args) -> TrainingArguments:
    if args.eval_mode != "none" or args.eval_strategy != "no":
        raise ValueError(
            "the first GRPO implementation does not run teacher-forced evaluation; "
            "use --eval-mode none --eval-strategy no"
        )
    if args.bf16 and args.fp16:
        raise ValueError("--bf16 and --fp16 cannot both be enabled")
    training_args = TrainingArguments(
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
        eval_strategy="no",
        dataloader_num_workers=args.dataloader_num_workers,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Keep complete model replicas on each DDP rank; disable FSDP sharding.
        fsdp="",
        fsdp_config=None,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        remove_unused_columns=False,
        label_names=[],
        bf16=args.bf16,
        fp16=args.fp16,
        tf32=args.tf32 and torch.cuda.is_available(),
        seed=args.seed,
        run_name=args.run_name,
        report_to=_trainer_report_to(args.report_to),
    )
    # Keep rollout-only settings on the TrainingArguments object because the
    # Trainer owns the optimizer/runtime while the parser owns CLI values.
    for name in (
        "num_generations",
        "rollout_batch_size",
        "max_prompt_length",
        "max_completion_length",
        "temperature",
        "top_p",
        "top_k",
        "clip_epsilon",
        "kl_coef",
        "iou_threshold",
    ):
        setattr(training_args, name, getattr(args, name))
    return training_args


def _reference_model(args):
    if not args.reference_model_name_or_path:
        return None
    from transformers import AutoModelForImageTextToText

    reference_path = Path(args.reference_model_name_or_path).expanduser()
    is_local_model = reference_path.is_dir()
    model_reference = (
        str(reference_path.resolve()) if is_local_model else args.reference_model_name_or_path
    )
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    reference = AutoModelForImageTextToText.from_pretrained(
        model_reference,
        cache_dir=args.cache_dir,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
        local_files_only=is_local_model,
    )
    reference.config.use_cache = False
    return reference


def _reward_weights(names: list[str], args) -> list[float]:
    weights = []
    for name in names:
        if name in {"box_iou", "iou", "accuracy_iou"}:
            weights.append(args.iou_reward_weight)
        elif name in {"score", "confidence", "accuracy_confidence"}:
            weights.append(args.score_reward_weight)
        elif name == "format":
            weights.append(args.format_reward_weight)
        else:
            weights.append(1.0)
    return weights


def train(args) -> None:
    _disable_proxy_environment()
    _validate_distributed_launch()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rollout_batch_size = _resolve_rollout_batch_size(args, world_size)
    if args.num_generations < 2:
        raise ValueError("rollout_n must be at least 2 for group-relative rewards")
    if args.max_completion_length < 1:
        raise ValueError("max_response_length must be a positive integer")
    logger.info(
        "GRPO rollout: prompt_batch=%d across %d ranks (%d prompts/rank), "
        "rollout_n=%d, max_response_length=%d",
        rollout_batch_size,
        world_size,
        args.per_device_train_batch_size,
        args.num_generations,
        args.max_completion_length,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    if args.bf16 and not torch.cuda.is_available():
        raise RuntimeError("bf16 training requires CUDA; pass --bf16 false for a CPU smoke test")
    unknown_rewards = sorted(set(args.reward_functions) - set(REWARD_FUNCS_REGISTRY))
    if unknown_rewards:
        available = ", ".join(sorted(REWARD_FUNCS_REGISTRY))
        raise ValueError(
            f"unknown reward function(s): {', '.join(unknown_rewards)}; choose from {available}"
        )
    if not any(weight > 0 for weight in _reward_weights(args.reward_functions, args)):
        raise ValueError("at least one reward weight must be positive")
    reward_weights = _reward_weights(args.reward_functions, args)
    if any(weight < 0 for weight in reward_weights):
        raise ValueError("reward weights must be non-negative")
    if not 0.0 <= args.iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")

    model, processor = load_model_and_processor(args)
    model.config.use_cache = False
    if (
        args.gradient_checkpointing
        and hasattr(model, "enable_input_require_grads")
    ):
        model.enable_input_require_grads()
    model = configure_trainable_parameters(model, args)

    data_args = data_config_from_args(args)
    dataset = GRPOPromptDataset(processor, data_args.dataset, data_args)
    training_args = _grpo_training_args(args)
    reference_model = _reference_model(args)
    _init_swanlab_if_requested(args)
    swanlab_enabled = any(
        item.lower() == "swanlab" for item in _report_to(args.report_to)
    )

    trainer = MultimodalGRPOTrainer(
        model=model,
        processor=processor,
        reward_funcs=args.reward_functions,
        reward_weights=reward_weights,
        args=training_args,
        train_dataset=dataset,
        reference_model=reference_model,
        swanlab_enabled=swanlab_enabled,
    )
    if args.stop_after_step is not None:
        if args.max_steps < args.stop_after_step:
            raise ValueError("stop-after-step must not exceed max-steps")
        trainer.add_callback(StopAfterStepCallback(args.stop_after_step))

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
    run_args = dict(vars(args))
    run_args["training_mode"] = "grpo"
    (output_dir / "run_args.json").write_text(
        json.dumps(run_args, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = build_grpo_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    train(args)


__all__ = ["main", "train"]
