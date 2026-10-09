from types import SimpleNamespace

import pytest

from qwen3vl_sft.train.arguments import build_grpo_parser
from qwen3vl_sft.train.grpo_runner import _resolve_rollout_batch_size


def test_rex_omni_rollout_argument_names_and_legacy_aliases():
    parser = build_grpo_parser()
    required = [
        "--dataset", "dummy.jsonl",
        "--model-name-or-path", "dummy-model",
        "--output-dir", "dummy-output",
    ]
    args = parser.parse_args(
        required
        + [
            "--rollout-batch-size", "4",
            "--rollout-n", "4",
            "--max-prompt-length", "4096",
            "--max-response-length", "512",
        ]
    )
    assert args.rollout_batch_size == 4
    assert args.num_generations == 4
    assert args.max_prompt_length == 4096
    assert args.max_completion_length == 512

    legacy = parser.parse_args(
        required
        + [
            "--num-generations", "4",
            "--max-completion-length", "512",
        ]
    )
    assert legacy.num_generations == 4
    assert legacy.max_completion_length == 512


def test_rollout_batch_size_resolves_global_prompts_to_per_rank_batch():
    args = SimpleNamespace(rollout_batch_size=4, per_device_train_batch_size=1)
    assert _resolve_rollout_batch_size(args, world_size=4) == 4
    assert args.per_device_train_batch_size == 1


def test_legacy_per_device_batch_derives_global_rollout_batch():
    args = SimpleNamespace(rollout_batch_size=None, per_device_train_batch_size=2)
    assert _resolve_rollout_batch_size(args, world_size=4) == 8
    assert args.rollout_batch_size == 8


def test_rollout_batch_must_divide_evenly_across_ranks():
    args = SimpleNamespace(rollout_batch_size=5, per_device_train_batch_size=1)
    with pytest.raises(ValueError, match="divisible"):
        _resolve_rollout_batch_size(args, world_size=4)



def test_grpo_checkpoint_defaults_to_every_1000_steps_and_can_be_overridden():
    parser = build_grpo_parser()
    required = [
        "--dataset", "dummy.jsonl",
        "--model-name-or-path", "dummy-model",
        "--output-dir", "dummy-output",
    ]
    defaults = parser.parse_args(required)
    assert defaults.save_strategy == "steps"
    assert defaults.save_steps == 1000

    custom = parser.parse_args(
        required + ["--save-strategy", "epoch", "--save-steps", "250"]
    )
    assert custom.save_strategy == "epoch"
    assert custom.save_steps == 250
