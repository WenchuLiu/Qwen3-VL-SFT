"""Training application and data pipeline.

The package is intentionally lazy at import time: lightweight tools such as
dataset builders can inspect argument defaults without importing PyTorch or
Transformers. The SFT entry point is ``python -m qwen3vl_sft.train``; the
score-aware GRPO entry point is ``python -m qwen3vl_sft.train.grpo``.
"""


def main() -> None:
    """Run the training CLI."""
    from .runner import main as _main

    _main()


def train(args) -> None:
    """Run one training job from a parsed argument namespace."""
    from .runner import train as _train

    _train(args)


__all__ = ["main", "train"]
