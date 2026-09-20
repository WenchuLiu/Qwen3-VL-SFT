"""Generation-based COCO evaluation shared by CLI and the training loop."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .metrics import evaluate_episode_predictions, trainer_metrics
from .protocol import (
    PROMPT_TEMPLATE_VERSION,
    PROTOCOL_NAME,
    build_eval_messages,
    load_category_descriptions,
)


def _resolve_media_path(value: object, base_path: Path) -> object:
    if not isinstance(value, str) or not value:
        return value
    if value.startswith(("http://", "https://", "file://", "data:image")):
        return value
    path = Path(value).expanduser()
    return str(path.resolve() if path.is_absolute() else (base_path / path).resolve())


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a local experiment input file."""
    return hashlib.sha256(Path(path).expanduser().resolve().read_bytes()).hexdigest()


def load_episodes(path: str | Path) -> tuple[dict, list[dict]]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        metadata = dict(payload)
        records = payload.get("records")
    else:
        metadata = {}
        records = payload
    if not isinstance(records, list) or not records:
        raise ValueError(f"evaluation file contains no episodes: {path}")
    if isinstance(metadata.get("query_annotations"), str):
        metadata["query_annotations"] = _resolve_media_path(
            metadata["query_annotations"], path.parent
        )
    normalized_records = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"episode {index} is not an object")
        if record.get("protocol") != PROTOCOL_NAME:
            raise ValueError(
                f"episode {index} uses unsupported protocol {record.get('protocol')!r}"
            )
        if (
            record.get("prompt_template_version")
            != PROMPT_TEMPLATE_VERSION
        ):
            raise ValueError(
                f"episode {index} uses a different prompt template; rebuild the episodes"
            )
        if not isinstance(record.get("support"), list) or not isinstance(
            record.get("query"), dict
        ):
            raise ValueError(f"episode {index} requires support and query fields")
        if not all(isinstance(frame, dict) for frame in record["support"]):
            raise ValueError(f"episode {index} support frames must be objects")
        normalized = dict(record)
        normalized["support"] = [
            {**frame, "image": _resolve_media_path(frame.get("image"), path.parent)}
            for frame in record["support"]
        ]
        normalized["query"] = {
            **record["query"],
            "image": _resolve_media_path(record["query"].get("image"), path.parent),
        }
        normalized_records.append(normalized)
    return metadata, normalized_records


def _move_inputs(inputs, device):
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def generate_responses(
    records: Sequence[dict],
    model,
    processor,
    device: str | torch.device,
    *,
    batch_size: int,
    min_pixels: int,
    max_pixels: int,
    max_new_tokens: int,
    visual_enhancement: bool = False,
    instruction_enhancement: bool = False,
    category_descriptions: Mapping[str, str] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[str]:
    """Generate deterministic responses for fixed episodes.

    ``visual_enhancement`` annotates support images with their GT boxes while
    leaving the query image untouched.  The flag is optional so the baseline
    prompt and all existing callers remain unchanged.  ``instruction_enhancement``
    adds the target category description to support and query questions without
    changing the images or model weights.
    """
    import torch

    if batch_size < 1 or max_new_tokens < 1:
        raise ValueError("batch_size and max_new_tokens must be positive")
    if min_pixels < 1 or max_pixels < min_pixels:
        raise ValueError("invalid image pixel budget")
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as error:
        raise ImportError(
            "generation evaluation requires qwen-vl-utils; install the project dependencies"
        ) from error

    tokenizer = processor.tokenizer
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model_was_training = model.training
    generation_config = getattr(model, "generation_config", None)
    original_generation_values = {}
    if generation_config is not None:
        for name in ("temperature", "top_p", "top_k"):
            if hasattr(generation_config, name):
                original_generation_values[name] = getattr(generation_config, name)
    model.eval()
    responses: list[str] = []
    try:
        for name in original_generation_values:
            setattr(generation_config, name, None)
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            messages_batch = [
                build_eval_messages(
                    record,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                    visual_enhancement=visual_enhancement,
                    instruction_enhancement=instruction_enhancement,
                    category_descriptions=category_descriptions,
                )
                for record in batch
            ]
            texts = [
                processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for messages in messages_batch
            ]
            image_inputs, video_inputs = process_vision_info(
                messages_batch,
                image_patch_size=int(processor.image_processor.patch_size),
            )
            inputs = processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                do_resize=False,
                return_tensors="pt",
            )
            inputs = _move_inputs(inputs, device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            prompt_length = inputs["input_ids"].shape[1]
            responses.extend(
                processor.batch_decode(
                    generated[:, prompt_length:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
            if progress_callback is not None:
                progress_callback(min(start + len(batch), len(records)), len(records))
    finally:
        for name, value in original_generation_values.items():
            setattr(generation_config, name, value)
        tokenizer.padding_side = original_padding_side
        model.train(model_was_training)
    return responses


def evaluate_loaded_model(
    records: Sequence[dict],
    model,
    processor,
    device: str | torch.device,
    *,
    batch_size: int,
    min_pixels: int,
    max_pixels: int,
    max_new_tokens: int,
    visual_enhancement: bool = False,
    instruction_enhancement: bool = False,
    category_descriptions: Mapping[str, str] | None = None,
    coco_annotations_path: str | None = None,
) -> dict:
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
        category_descriptions=category_descriptions,
    )
    return evaluate_episode_predictions(
        records,
        responses,
        coco_annotations_path=coco_annotations_path,
    )


def result_payload(
    result: dict,
    *,
    model_path: str,
    adapter_path: str | None,
    episodes_path: str,
    device: str,
    batch_size: int,
    min_pixels: int,
    max_pixels: int,
    max_new_tokens: int,
    visual_enhancement: bool = False,
    instruction_enhancement: bool = False,
    category_descriptions_path: str | None = None,
    category_descriptions_sha256: str | None = None,
    coco_annotations_path: str | None = None,
    runtime_seconds: float | None = None,
) -> dict:
    episodes_file = Path(episodes_path)
    episodes_hash = (
        hashlib.sha256(episodes_file.read_bytes()).hexdigest()
        if episodes_file.is_file()
        else None
    )
    if visual_enhancement and instruction_enhancement:
        prompt_variant = "visual_and_instruction_enhanced"
    elif visual_enhancement:
        prompt_variant = "visual_enhanced"
    elif instruction_enhancement:
        prompt_variant = "instruction_enhanced"
    else:
        prompt_variant = "baseline"
    return {
        "protocol": PROTOCOL_NAME,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "model_path": model_path,
        "adapter_path": adapter_path,
        "episodes_path": episodes_path,
        "episodes_sha256": episodes_hash,
        "device": device,
        "batch_size": batch_size,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "max_new_tokens": max_new_tokens,
        "visual_enhancement": bool(visual_enhancement),
        # Keep the short name used by the original VE evaluator available for
        # downstream result consumers.
        "ve": bool(visual_enhancement),
        "instruction_enhancement": bool(instruction_enhancement),
        # Keep a short compatibility key alongside the descriptive name.
        "ie": bool(instruction_enhancement),
        "prompt_variant": prompt_variant,
        "category_descriptions_path": category_descriptions_path,
        "category_descriptions_sha256": category_descriptions_sha256,
        "coco_annotations_path": coco_annotations_path,
        "runtime_seconds": runtime_seconds,
        **result,
    }


def evaluate_checkpoint(
    *,
    model_path: str,
    adapter_path: str | None,
    episodes_path: str,
    device: str,
    batch_size: int,
    min_pixels: int,
    max_pixels: int,
    max_new_tokens: int,
    attention: str,
    visual_enhancement: bool = False,
    instruction_enhancement: bool = False,
    category_descriptions_path: str | None = None,
) -> dict:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    category_descriptions = None
    description_path = None
    description_hash = None
    if category_descriptions_path is not None:
        description_path = str(Path(category_descriptions_path).expanduser().resolve())
        category_descriptions = load_category_descriptions(description_path)
        description_hash = file_sha256(description_path)

    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=device,
        attn_implementation=attention,
        trust_remote_code=True,
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    metadata, records = load_episodes(episodes_path)
    coco_annotations_path = metadata.get("query_annotations")
    started = time.monotonic()
    result = evaluate_loaded_model(
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
        category_descriptions=category_descriptions,
        coco_annotations_path=coco_annotations_path,
    )
    return result_payload(
        result,
        model_path=model_path,
        adapter_path=adapter_path,
        episodes_path=str(Path(episodes_path).resolve()),
        device=device,
        batch_size=batch_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        max_new_tokens=max_new_tokens,
        visual_enhancement=visual_enhancement,
        instruction_enhancement=instruction_enhancement,
        category_descriptions_path=description_path,
        category_descriptions_sha256=description_hash,
        coco_annotations_path=coco_annotations_path,
        runtime_seconds=time.monotonic() - started,
    )
