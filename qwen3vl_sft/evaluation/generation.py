"""Backward-compatible module alias for :mod:`evaluation.coco.generation`."""

import sys as _sys

from .coco import generation as _implementation

_sys.modules[__name__] = _implementation
