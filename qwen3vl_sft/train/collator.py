"""Batch collation for multimodal supervised fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from ..data.processing import IGNORE_INDEX


@dataclass
class MultimodalDataCollator:
    """Pad text fields while preserving the unpadded vision batches."""

    tokenizer: object
    model_max_length: int

    def __call__(self, instances: Sequence[Mapping[str, object]]) -> dict[str, object]:
        if not instances:
            raise ValueError("cannot collate an empty batch")

        input_ids = []
        labels = []
        positions = []
        for instance in instances:
            ids = instance["input_ids"]
            target = instance["labels"]
            position = instance["position_ids"]
            if ids.shape[-1] > self.model_max_length:
                raise ValueError(
                    "a sample exceeds model_max_length; increase --model-max-length "
                    "because truncating multimodal token sequences can desynchronize "
                    "image/video tokens and their pixel grids"
                )
            input_ids.append(ids.squeeze(0))
            labels.append(target.squeeze(0))
            positions.append(position)

        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, "eos_token_id", 0)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        max_length = input_ids.shape[1]
        padded_positions = [
            torch.nn.functional.pad(
                position[..., :max_length],
                (0, max_length - position.shape[-1]),
                value=1,
            )
            for position in positions
        ]
        batch: dict[str, object] = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(pad_token_id),
            "position_ids": torch.cat(padded_positions, dim=1),
        }

        for input_key in (
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
        ):
            values = [instance.get(input_key) for instance in instances]
            values = [value for value in values if value is not None]
            if values:
                batch[input_key] = torch.cat(values, dim=0)
        return batch
