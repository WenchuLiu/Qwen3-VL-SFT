"""Construction of the train/eval data module consumed by Trainer."""

from __future__ import annotations

from .collator import MultimodalDataCollator
from .dataset import SupervisedDataset, split_dataset


def make_data_module(processor, data_args) -> dict:
    """Build datasets and the collator from the parsed data configuration."""
    train_dataset = SupervisedDataset(processor, data_args.dataset, data_args)
    eval_dataset = None
    if data_args.eval_mode == "loss":
        if data_args.eval_dataset:
            eval_dataset = SupervisedDataset(processor, data_args.eval_dataset, data_args)
        else:
            train_dataset, eval_dataset = split_dataset(
                train_dataset,
                data_args.eval_ratio,
                data_args.eval_seed,
            )
    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": MultimodalDataCollator(
            tokenizer=processor.tokenizer,
            model_max_length=data_args.model_max_length,
        ),
    }
