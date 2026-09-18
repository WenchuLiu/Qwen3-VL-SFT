"""Backward-compatible model API.

New code should import from :mod:`qwen3vl_sft.model.loader`, which follows the
same package boundary as the reference LocateAnything project.
"""

from .model.loader import (
    configure_trainable_parameters,
    load_model_and_processor,
    save_model_and_processor,
)

__all__ = [
    "configure_trainable_parameters",
    "load_model_and_processor",
    "save_model_and_processor",
]
