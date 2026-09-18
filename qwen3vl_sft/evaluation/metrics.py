"""Backward-compatible module alias for :mod:`evaluation.coco.metrics`."""

import sys as _sys

from .coco import metrics as _implementation

_sys.modules[__name__] = _implementation
