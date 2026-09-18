"""Backward-compatible alias for :mod:`qwen3vl_sft.train.trainer`.

The alias is installed at module level so patches against the historical
``qwen3vl_sft.trainers`` path still modify the canonical implementation.
"""

import sys as _sys

from .train import trainer as _implementation

_sys.modules[__name__] = _implementation
