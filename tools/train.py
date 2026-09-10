#!/usr/bin/env python3
"""Repository launcher for ``python tools/train.py ...``."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen3vl_sft.train import main  # noqa: E402


if __name__ == "__main__":
    main()
