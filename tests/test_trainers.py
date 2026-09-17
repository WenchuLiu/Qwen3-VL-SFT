import unittest
from types import SimpleNamespace
from unittest.mock import patch

from qwen3vl_sft.trainers import TrainingTelemetryCallback, format_iter_epoch


class TrainerTelemetryTest(unittest.TestCase):
    def test_iter_epoch_uses_mmdetection_style(self):
        self.assertEqual(format_iter_epoch(74, 0.05, 12), "74[1/12]")
        self.assertEqual(format_iter_epoch(17748, 12.0, 12), "17748[12/12]")

    def test_training_log_does_not_require_swanlab(self):
        callback = TrainingTelemetryCallback()
        args = SimpleNamespace(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            world_size=2,
            dataloader_num_workers=2,
            num_train_epochs=12,
            report_to=[],
        )
        state = SimpleNamespace(
            is_world_process_zero=True,
            global_step=10,
            max_steps=17748,
            epoch=0.01,
        )
        with patch("qwen3vl_sft.trainers._nvidia_smi_metrics", return_value={}):
            callback.on_train_begin(args, state, object())
            self.assertIsNotNone(
                callback.on_log(
                    args,
                    state,
                    object(),
                    logs={"loss": 0.9, "grad_norm": 1.2, "learning_rate": 1e-4},
                )
            )

    def test_swanlab_payload_contains_only_requested_metrics(self):
        callback = TrainingTelemetryCallback(swanlab_enabled=True)
        args = SimpleNamespace(
            report_to=[],
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            world_size=1,
            dataloader_num_workers=0,
            num_train_epochs=1,
        )
        state = SimpleNamespace(
            is_world_process_zero=True,
            global_step=10,
            max_steps=100,
            epoch=0.1,
        )
        logs = {
            "loss": 0.9,
            "grad_norm": 1.2,
            "learning_rate": 1e-4,
            "eval_coco_0shot_f1": 0.1,
            "eval_coco_0shot_map": 0.2,
            "eval_coco_1shot_f1": 0.3,
            "eval_coco_1shot_map": 0.4,
            "eval_coco_2shot_f1": 0.5,
            "eval_coco_2shot_map": 0.6,
            "eval_coco_4shot_f1": 0.7,
            "eval_coco_4shot_map": 0.8,
            "eval_coco_1shot_count_accuracy": 0.9,
            "throughput/steps_per_second": 10.0,
        }
        expected = {
            key: value
            for key, value in logs.items()
            if key in {
                "loss",
                "grad_norm",
                "learning_rate",
                "eval_coco_0shot_f1",
                "eval_coco_0shot_map",
                "eval_coco_1shot_f1",
                "eval_coco_1shot_map",
                "eval_coco_2shot_f1",
                "eval_coco_2shot_map",
                "eval_coco_4shot_f1",
                "eval_coco_4shot_map",
            }
        }
        with (
            patch.object(callback, "_log_swanlab") as log_swanlab,
            patch("qwen3vl_sft.trainers._nvidia_smi_metrics", return_value={}),
        ):
            callback.on_log(args, state, object(), logs=logs)
        log_swanlab.assert_called_once_with(expected, 10)


if __name__ == "__main__":
    unittest.main()
