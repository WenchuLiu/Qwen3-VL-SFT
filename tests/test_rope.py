import unittest

import torch

from qwen3vl_sft.data.rope import get_qwen3_rope_index


class RopeTest(unittest.TestCase):
    def test_video_grid_is_split_into_one_frame_grids(self):
        input_ids = torch.tensor(
            [[
                10,
                151652,
                151656,
                151656,
                151656,
                151656,
                11,
                151652,
                151656,
                151656,
                151656,
                151656,
                12,
            ]]
        )
        position_ids, deltas = get_qwen3_rope_index(
            spatial_merge_size=2,
            input_ids=input_ids,
            video_grid_thw=torch.tensor([[2, 4, 4]]),
        )

        self.assertEqual(position_ids.shape, (3, 1, 13))
        self.assertEqual(deltas.shape, (1, 1))
        self.assertEqual(position_ids[0, 0, 2:6].tolist(), [2, 2, 2, 2])
        self.assertEqual(position_ids[0, 0, 8:12].tolist(), [6, 6, 6, 6])

    def test_padding_uses_attention_mask_for_text_only_rows(self):
        input_ids = torch.tensor([[10, 11, 0, 0], [20, 21, 22, 0]])
        attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
        position_ids, deltas = get_qwen3_rope_index(
            spatial_merge_size=2,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        self.assertEqual(position_ids[:, 0].tolist(), [[0, 1, 1, 1]] * 3)
        self.assertEqual(position_ids[:, 1].tolist(), [[0, 1, 2, 1]] * 3)
        self.assertEqual(deltas.tolist(), [[-2], [-1]])


if __name__ == "__main__":
    unittest.main()
