"""Qwen3-VL loading and trainable-parameter configuration.

The model package owns Hugging Face and PEFT integration.  Dataset and
evaluation code should depend on the processor interface instead of importing
model-loading details.
"""

from __future__ import annotations

from pathlib import Path

import torch


def load_model_and_processor(args):
    """Load a Qwen3-VL checkpoint and configure its processor."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    adapter_path = getattr(args, "adapter_path", None)
    base_path = Path(args.model_name_or_path).expanduser()
    adapter_reference = adapter_path
    if adapter_path and not args.lora_enable:
        raise ValueError("--adapter-path requires --lora-enable true")
    if (
        adapter_path
        and base_path.is_dir()
        and (base_path / "adapter_config.json").is_file()
    ):
        raise ValueError(
            "--model-name-or-path must point to the base model when --adapter-path is set"
        )
    if adapter_path:
        local_adapter_path = Path(adapter_path).expanduser()
        if local_adapter_path.exists() and not local_adapter_path.is_dir():
            raise NotADirectoryError(f"adapter path is not a directory: {local_adapter_path}")
        if local_adapter_path.is_dir():
            if not (local_adapter_path / "adapter_config.json").is_file():
                raise FileNotFoundError(
                    f"adapter_config.json not found in adapter directory: {local_adapter_path}"
                )
            adapter_reference = str(local_adapter_path.resolve())

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    if adapter_reference:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            adapter_reference,
            is_trainable=True,
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
    """Apply either the LoRA policy or the explicit full-tuning policy."""
    adapter_path = getattr(args, "adapter_path", None)
    if adapter_path and not args.lora_enable:
        raise ValueError("--adapter-path requires --lora-enable true")

    if args.lora_enable:
        if adapter_path:
            if not getattr(model, "peft_config", None):
                raise ValueError("--adapter-path was set, but the model has no loaded PEFT adapter")
            trainable = sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            if trainable == 0:
                raise ValueError(
                    "the loaded adapter has no trainable parameters; it must be loaded with is_trainable=True"
                )
            model.print_trainable_parameters()
            return model

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
    """Save the Trainer model and the matching processor together."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output))
    processor.save_pretrained(str(output))
