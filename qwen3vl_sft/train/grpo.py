"""Command-line entry point for score-aware multimodal GRPO."""

from .grpo_runner import main, train
from .rewards import (
    REWARD_FUNCS_REGISTRY,
    accuracy_reward_confidence,
    accuracy_reward_iou,
    confidence_reward,
    format_reward,
    iou_reward,
    reward_funcs_registry,
    score_reward,
)

__all__ = [
    "REWARD_FUNCS_REGISTRY",
    "accuracy_reward_confidence",
    "accuracy_reward_iou",
    "confidence_reward",
    "format_reward",
    "iou_reward",
    "main",
    "reward_funcs_registry",
    "score_reward",
    "train",
]


if __name__ == "__main__":
    main()
