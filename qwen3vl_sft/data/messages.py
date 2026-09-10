"""Convert the public Qwen VL conversation schema to HF chat messages."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping


_MEDIA_PATTERN = re.compile(r"(<image>|<video>)")


def _resolve_path(value: object, base_path: Path) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("media paths must be non-empty strings")
    path = Path(value).expanduser()
    return str(path.resolve() if path.is_absolute() else (base_path / path).resolve())


def build_messages(record: Mapping[str, object], *, base_path: str | Path = ".") -> list[dict]:
    """Build HF multimodal messages and verify placeholder/media alignment."""
    base_path = Path(base_path).expanduser()
    images = record.get("image", [])
    videos = record.get("video", [])
    if isinstance(images, str):
        images = [images]
    if isinstance(videos, str):
        videos = [videos]
    if not isinstance(images, list) or not isinstance(videos, list):
        raise ValueError("'image' and 'video' must be a path or a list of paths")

    image_pool = [{"type": "image", "image": _resolve_path(path, base_path)} for path in images]
    video_pool = [{"type": "video", "video": _resolve_path(path, base_path)} for path in videos]
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("record requires a non-empty conversations list")

    messages: list[dict] = []
    for turn_index, turn in enumerate(conversations):
        if not isinstance(turn, Mapping):
            raise ValueError(f"conversation turn {turn_index} must be an object")
        source_role = turn.get("from")
        if source_role in {"human", "user"}:
            role = "user"
        elif source_role in {"gpt", "assistant"}:
            role = "assistant"
        elif source_role == "system":
            role = "system"
        else:
            raise ValueError(f"unsupported conversation role: {source_role!r}")
        text = turn.get("value")
        if not isinstance(text, str):
            raise ValueError(f"conversation turn {turn_index} value must be text")

        if role != "user":
            if "<image>" in text or "<video>" in text:
                raise ValueError("media placeholders are only valid in user turns")
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})
            continue

        content = []
        for part in _MEDIA_PATTERN.split(text):
            if part == "<image>":
                if not image_pool:
                    raise ValueError("more <image> placeholders than image paths")
                content.append(image_pool.pop(0))
            elif part == "<video>":
                if not video_pool:
                    raise ValueError("more <video> placeholders than video paths")
                content.append(video_pool.pop(0))
            elif part.strip():
                content.append({"type": "text", "text": part.strip()})
        if not content:
            raise ValueError(f"conversation turn {turn_index} has no content")
        messages.append({"role": role, "content": content})

    if image_pool:
        raise ValueError(f"{len(image_pool)} image path(s) were not referenced by <image>")
    if video_pool:
        raise ValueError(f"{len(video_pool)} video path(s) were not referenced by <video>")
    return messages
