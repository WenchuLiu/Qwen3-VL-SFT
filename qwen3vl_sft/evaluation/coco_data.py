"""Backward-compatible module alias for :mod:`evaluation.coco.data`."""

import sys as _sys

from .coco import data as _implementation

_sys.modules[__name__] = _implementation
