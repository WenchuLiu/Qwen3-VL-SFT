"""Portable resolution of local media paths stored in dataset manifests."""

from __future__ import annotations

from pathlib import Path

_REMOTE_PREFIXES = ("http://", "https://", "file://", "data:")
_COCO_LAYOUT_PREFIXES = ("", "COCO", "coco", "COCO2017", "COCO2017train", "COCO2017val")
_COCO_PATH_MARKERS = frozenset({"annotations", "train2017", "val2017"})


def _candidate_roots(base_path: Path) -> list[Path]:
    roots: list[Path] = []
    current = base_path.expanduser().resolve()
    for _ in range(4):
        if current not in roots:
            roots.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    return roots


def _coco_suffix(path: Path) -> Path | None:
    for index, component in enumerate(path.parts):
        if component.lower() in _COCO_PATH_MARKERS:
            return Path(*path.parts[index:])
    return None


def _candidate_paths(path: Path, base_path: Path) -> list[Path]:
    roots = _candidate_roots(base_path)
    candidates: list[Path] = []
    if path.is_absolute():
        suffix = _coco_suffix(path)
        if suffix is None:
            return candidates
        for root in roots:
            for prefix in _COCO_LAYOUT_PREFIXES:
                candidate = root / prefix / suffix if prefix else root / suffix
                candidate = candidate.resolve()
                if candidate not in candidates:
                    candidates.append(candidate)
        return candidates

    for root in roots:
        candidate = (root / path).resolve()
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def resolve_media_path(value: object, base_path: str | Path = ".") -> object:
    """Resolve a manifest media path and relocate stale COCO absolute paths.

    Manifests generated on another host may contain an absolute path into a
    ModelScope extraction cache.  Existing absolute paths remain unchanged;
    missing local paths are looked up under ``base_path`` and its nearby
    parents using the standard COCO directory layout.
    """
    if not isinstance(value, str) or not value:
        return value
    if value.startswith(_REMOTE_PREFIXES):
        return value

    path = Path(value).expanduser()
    base_path = Path(base_path).expanduser()
    candidates = [path.resolve()] if path.is_absolute() else []
    candidates.extend(_candidate_paths(path, base_path))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    tried = ", ".join(str(candidate) for candidate in candidates[:8])
    if not tried:
        tried = "no portable COCO suffix found"
    raise FileNotFoundError(
        f"media file does not exist: {value!r}. Set DATA_ROOT to a directory "
        "containing COCO/train2017 and COCO/val2017, or provide a valid local "
        f"media path. Tried: {tried}"
    )
