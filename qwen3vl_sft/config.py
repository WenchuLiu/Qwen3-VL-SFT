"""Backward-compatible configuration imports.

The canonical parser lives in the ``train/arguments.py`` module, following
the package layout used by the reference LocateAnything project.
"""

from .train.arguments import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    DataConfig,
    add_data_arguments,
    add_training_arguments,
    build_train_parser,
    data_config_from_args,
    str_to_bool,
)

__all__ = [
    "DEFAULT_MAX_PIXELS",
    "DEFAULT_MIN_PIXELS",
    "DataConfig",
    "add_data_arguments",
    "add_training_arguments",
    "build_train_parser",
    "data_config_from_args",
    "str_to_bool",
]
