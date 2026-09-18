"""Backward-compatible module alias for :mod:`evaluation.coco.protocol`."""

import sys as _sys

from .coco import protocol as _implementation

_sys.modules[__name__] = _implementation
