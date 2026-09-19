# Shell launchers

This directory is compatibility-only. The canonical reusable scripts remain in
`scripts/`; these wrappers delegate to them without duplicating experiment
logic. New commands should be added under `scripts/`, not here.

This includes `shell/train_grpo.sh`, the wrapper for score-aware multimodal
GRPO training.
