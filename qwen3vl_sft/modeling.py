"""Qwen3-VL model loading and trainable-parameter configuration."""

from __future__ import annotations

from pathlib import Path

import torch


def load_model_and_processor(args):
    """Load a Qwen3-VL checkpoint with lazy imports for lightweight tooling."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    processor.tokenizer.model_max_length = args.model_max_length
    processor.tokenizer.padding_side = "right"
    return model, processor


def configure_trainable_parameters(model, args):
    if args.lora_enable:
        if args.lora_r < 1 or args.lora_alpha < 1:
            raise ValueError("lora-r and lora-alpha must be positive")
        if not 0 <= args.lora_dropout < 1:
            raise ValueError("lora-dropout must be in [0, 1)")
        from peft import LoraConfig, TaskType, get_peft_model

        for parameter in model.parameters():
            parameter.requires_grad = False
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            ),
        )
        model.print_trainable_parameters()
        return model

    visual = getattr(model, "visual", None)
    if visual is None:
        raise AttributeError("Qwen3-VL model does not expose a visual module")
    language_model = getattr(model, "language_model", None)
    if language_model is None:
        raise AttributeError("Qwen3-VL model does not expose language_model")
    for parameter in model.parameters():
        parameter.requires_grad = False
    if args.tune_mm_vision:
        for parameter in visual.parameters():
            parameter.requires_grad = True
    if args.tune_mm_mlp:
        merger = getattr(visual, "merger", None)
        if merger is None:
            raise AttributeError("Qwen3-VL visual module does not expose merger")
        for parameter in merger.parameters():
            parameter.requires_grad = True
    if args.tune_mm_llm:
        for parameter in language_model.parameters():
            parameter.requires_grad = True
        if getattr(model, "lm_head", None) is not None:
            for parameter in model.lm_head.parameters():
                parameter.requires_grad = True
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable == 0:
        raise ValueError(
            "no trainable parameters selected; enable LoRA or one of "
            "--tune-mm-vision/--tune-mm-mlp/--tune-mm-llm"
        )
    print(f"trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    return model


def save_model_and_processor(trainer, processor, output_dir: str) -> None:
    """Save PEFT adapters or the selected full model without CPU duplication."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output))
    processor.save_pretrained(str(output))
