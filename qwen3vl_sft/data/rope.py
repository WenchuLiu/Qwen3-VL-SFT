"""Qwen3-VL multimodal RoPE position IDs.

This mirrors ``Qwen3VLModel.get_rope_index`` from Transformers.  It is kept
local because preprocessing happens before the model is handed to Trainer, but
the model's token IDs remain configurable for future model revisions.
"""

from __future__ import annotations

from typing import Optional

import torch


def get_qwen3_rope_index(
    *,
    spatial_merge_size: int,
    input_ids: torch.LongTensor,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    image_token_id: int = 151655,
    video_token_id: int = 151656,
    vision_start_token_id: int = 151652,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return position IDs for Qwen3-VL image/video and text tokens.

    Qwen3-VL represents video time with timestamp tokens in the rendered
    prompt.  Therefore video grids are split into one-frame grids here and the
    temporal coordinate of every visual patch is zero, exactly as in the
    upstream model implementation.
    """
    if input_ids is None or input_ids.ndim != 2:
        raise ValueError("input_ids must be a [batch, sequence] tensor")

    if video_grid_thw is not None:
        if video_grid_thw.ndim != 2 or video_grid_thw.shape[-1] != 3:
            raise ValueError("video_grid_thw must have shape [num_videos, 3]")
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw = video_grid_thw.clone()
        video_grid_thw[:, 0] = 1

    if image_grid_thw is not None and (
        image_grid_thw.ndim != 2 or image_grid_thw.shape[-1] != 3
    ):
        raise ValueError("image_grid_thw must have shape [num_images, 3]")

    if image_grid_thw is None and video_grid_thw is None:
        if attention_mask is not None:
            attention_mask = attention_mask.to(input_ids.device)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            max_position_ids = position_ids.max(dim=0).values.max(dim=-1, keepdim=True).values
            deltas = max_position_ids + 1 - attention_mask.shape[-1]
            return position_ids, deltas
        position_ids = (
            torch.arange(input_ids.shape[1], device=input_ids.device)
            .view(1, 1, -1)
            .expand(3, input_ids.shape[0], -1)
        )
        return position_ids, torch.zeros(
            (input_ids.shape[0], 1), device=input_ids.device, dtype=input_ids.dtype
        )

    total_input_ids = input_ids
    if attention_mask is None:
        attention_mask = torch.ones_like(total_input_ids)
    elif attention_mask.shape != total_input_ids.shape:
        raise ValueError("attention_mask must have the same shape as input_ids")
    position_ids = torch.ones(
        3,
        input_ids.shape[0],
        input_ids.shape[1],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    image_index, video_index = 0, 0
    deltas = []
    attention_mask = attention_mask.to(total_input_ids.device)

    for batch_index, row in enumerate(total_input_ids):
        valid_mask = attention_mask[batch_index] == 1
        valid_row = row[valid_mask]
        vision_starts = torch.argwhere(valid_row == vision_start_token_id).flatten()
        vision_starts = vision_starts[vision_starts + 1 < valid_row.numel()]
        vision_tokens = valid_row[vision_starts + 1]
        image_count = int((vision_tokens == image_token_id).sum().item())
        video_count = int((vision_tokens == video_token_id).sum().item())
        row_tokens = valid_row.tolist()
        position_chunks = []
        start = 0
        remaining_images, remaining_videos = image_count, video_count

        for _ in range(image_count + video_count):
            image_end = (
                row_tokens.index(image_token_id, start)
                if remaining_images and image_token_id in row_tokens[start:]
                else len(row_tokens) + 1
            )
            video_end = (
                row_tokens.index(video_token_id, start)
                if remaining_videos and video_token_id in row_tokens[start:]
                else len(row_tokens) + 1
            )
            if image_end < video_end:
                if image_grid_thw is None or image_index >= len(image_grid_thw):
                    raise ValueError("input contains more images than image_grid_thw entries")
                t, h, w = image_grid_thw[image_index]
                image_index += 1
                remaining_images -= 1
                end = image_end
            elif video_end < image_end:
                if video_grid_thw is None or video_index >= len(video_grid_thw):
                    raise ValueError("input contains more videos than video_grid_thw entries")
                t, h, w = video_grid_thw[video_index]
                video_index += 1
                remaining_videos -= 1
                end = video_end
            else:
                raise ValueError("vision token count does not match the supplied media grids")

            grid_t = int(t.item())
            grid_h = int(h.item()) // spatial_merge_size
            grid_w = int(w.item()) // spatial_merge_size
            if grid_t < 1 or grid_h < 1 or grid_w < 1:
                raise ValueError(
                    "media grid dimensions must be positive and divisible by merge size"
                )
            text_length = end - start
            start_position = (
                position_chunks[-1].max() + 1 if position_chunks else 0
            )
            position_chunks.append(
                torch.arange(text_length, device=input_ids.device)
                .view(1, -1)
                .expand(3, -1)
                + start_position
            )

            time_index = (
                torch.arange(grid_t, device=input_ids.device)
                .view(-1, 1)
                .expand(-1, grid_h * grid_w)
                .flatten()
            )
            height_index = (
                torch.arange(grid_h, device=input_ids.device)
                .view(1, -1, 1)
                .expand(grid_t, -1, grid_w)
                .flatten()
            )
            width_index = (
                torch.arange(grid_w, device=input_ids.device)
                .view(1, 1, -1)
                .expand(grid_t, grid_h, -1)
                .flatten()
            )
            position_chunks.append(
                torch.stack((time_index, height_index, width_index))
                + text_length
                + start_position
            )
            start = end + grid_t * grid_h * grid_w

        if start < len(row_tokens):
            start_position = position_chunks[-1].max() + 1 if position_chunks else 0
            text_length = len(row_tokens) - start
            position_chunks.append(
                torch.arange(text_length, device=input_ids.device)
                .view(1, -1)
                .expand(3, -1)
                + start_position
            )
        row_positions = torch.cat(position_chunks, dim=1).reshape(3, -1)
        if row_positions.shape[-1] != int(valid_mask.sum().item()):
            raise ValueError("computed RoPE positions do not match the unpadded sequence")
        position_ids[..., batch_index, valid_mask] = row_positions.to(position_ids.device)
        deltas.append(row_positions.max() + 1 - len(total_input_ids[batch_index]))

    return position_ids, torch.stack(deltas).to(input_ids.device).unsqueeze(1)
