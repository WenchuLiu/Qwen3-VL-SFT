import unittest

from qwen3vl_sft.config import build_train_parser


class TrainConfigTest(unittest.TestCase):
    def test_coco_training_defaults_to_twelve_epochs_and_swanlab(self):
        args = build_train_parser().parse_args(
            [
                "--model-name-or-path",
                "model",
                "--dataset",
                "train.json",
                "--output-dir",
                "output",
            ]
        )

        self.assertEqual(args.num_train_epochs, 12.0)
        self.assertEqual(args.report_to, "swanlab")


if __name__ == "__main__":
    unittest.main()
