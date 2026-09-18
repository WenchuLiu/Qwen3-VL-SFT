"""Validation for the public Qwen VL conversation record schema."""

from __future__ import annotations

from collections.abc import Mapping


def validate_sft_record(record: Mapping[str, object], *, index: int = 0) -> None:
    """Fail early when a generated or hand-written SFT record is malformed."""
    required = ("image", "conversations")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"record {index} is missing fields: {', '.join(missing)}")

    images = record["image"]
    conversations = record["conversations"]
    if not isinstance(images, (str, list)) or not isinstance(conversations, list):
        raise ValueError(f"record {index} has invalid image/conversations fields")
    if not conversations:
        raise ValueError(f"record {index} has no conversation turns")

    for turn_index, turn in enumerate(conversations):
        if not isinstance(turn, dict) or not isinstance(turn.get("from"), str):
            raise ValueError(f"record {index} turn {turn_index} is malformed")
        if not isinstance(turn.get("value"), str):
            raise ValueError(f"record {index} turn {turn_index} value must be text")

