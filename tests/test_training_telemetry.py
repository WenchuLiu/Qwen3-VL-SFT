import importlib
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def _load_trainer_without_optional_training_dependencies():
    """Load the telemetry callback while replacing unrelated heavy imports."""
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        memory_allocated=lambda: 0,
        memory_reserved=lambda: 0,
        max_memory_allocated=lambda: 0,
    )
    torch.distributed = types.SimpleNamespace(
        is_available=lambda: False,
        is_initialized=lambda: False,
    )

    class Trainer:
        pass

    class TrainerCallback:
        pass

    class PrinterCallback:
        pass

    class ProgressCallback:
        pass

    transformers = types.ModuleType("transformers")
    transformers.Trainer = Trainer
    trainer_callback = types.ModuleType("transformers.trainer_callback")
    trainer_callback.PrinterCallback = PrinterCallback
    trainer_callback.ProgressCallback = ProgressCallback
    trainer_callback.TrainerCallback = TrainerCallback
    trainer_utils = types.ModuleType("transformers.trainer_utils")
    trainer_utils.speed_metrics = lambda *args, **kwargs: {}

    generation = types.ModuleType("qwen3vl_sft.evaluation.coco.generation")
    generation.generate_responses = None
    generation.load_episodes = None
    metrics = types.ModuleType("qwen3vl_sft.evaluation.coco.metrics")
    metrics.evaluate_episode_predictions = None
    metrics.trainer_metrics = None

    fake_modules = {
        "torch": torch,
        "transformers": transformers,
        "transformers.trainer_callback": trainer_callback,
        "transformers.trainer_utils": trainer_utils,
        "qwen3vl_sft.evaluation.coco.generation": generation,
        "qwen3vl_sft.evaluation.coco.metrics": metrics,
    }
    sys.modules.pop("qwen3vl_sft.train.trainer", None)
    with patch.dict(sys.modules, fake_modules):
        return importlib.import_module("qwen3vl_sft.train.trainer")


class TrainingTelemetryTest(unittest.TestCase):
    def test_rank_responses_are_restored_to_manifest_order(self):
        trainer = _load_trainer_without_optional_training_dependencies()
        merge = getattr(trainer, "_merge_ranked_responses", lambda *_: None)

        responses = merge(
            [
                {"rank": 0, "responses": ["response-0", "response-4"], "error": None},
                {"rank": 1, "responses": ["response-1", "response-5"], "error": None},
                {"rank": 2, "responses": ["response-2", "response-6"], "error": None},
                {"rank": 3, "responses": ["response-3", "response-7"], "error": None},
            ],
            8,
        )

        self.assertEqual(responses, [f"response-{index}" for index in range(8)])

    def test_generation_eval_runs_each_shot_in_order_without_writing_results(self):
        trainer = _load_trainer_without_optional_training_dependencies()
        generated_shots = []
        logged_metrics = []

        class Model:
            config = types.SimpleNamespace(use_cache=False)

            @staticmethod
            def parameters():
                return iter([types.SimpleNamespace(device="cuda:0")])

        def generate(records, *args, **kwargs):
            del args, kwargs
            shot = records[0]["num_shots"]
            generated_shots.append(shot)
            return [f"{shot}-shot-response-{index}" for index in range(len(records))]

        def evaluate_predictions(records, responses, **kwargs):
            del responses, kwargs
            return {"shot": records[0]["num_shots"]}

        def flatten_metrics(result):
            shot = result["shot"]
            return {
                f"eval_coco_{shot}shot_f1": shot + 0.25,
                f"eval_coco_{shot}shot_map": shot + 0.125,
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator = object.__new__(trainer.GenerationEvalTrainer)
            evaluator.accelerator = types.SimpleNamespace(unwrap_model=lambda model: model)
            evaluator.model = Model()
            evaluator.args = types.SimpleNamespace(output_dir=str(root))
            evaluator.state = types.SimpleNamespace(global_step=370)
            evaluator.control = None
            evaluator.callback_handler = types.SimpleNamespace(
                on_evaluate=lambda args, state, control, metrics: control
            )
            evaluator.log = lambda values: logged_metrics.append(values)
            evaluator.generation_eval_tasks = [
                {
                    "shot": 0,
                    "records": [{"num_shots": 0}, {"num_shots": 0}],
                    "metadata": {},
                },
                {
                    "shot": 1,
                    "records": [{"num_shots": 1}, {"num_shots": 1}],
                    "metadata": {},
                },
            ]
            evaluator.generation_eval_processor = object()
            evaluator.generation_eval_batch_size = 1
            evaluator.generation_eval_min_pixels = 4096
            evaluator.generation_eval_max_pixels = 640000
            evaluator.generation_eval_max_new_tokens = 1024
            evaluator.generation_eval_visual_enhancement = False

            with (
                patch.object(
                    trainer,
                    "generate_responses",
                    side_effect=generate,
                    create=True,
                ),
                patch.object(
                    trainer,
                    "evaluate_episode_predictions",
                    side_effect=evaluate_predictions,
                    create=True,
                ),
                patch.object(trainer, "trainer_metrics", side_effect=flatten_metrics),
            ):
                metrics = evaluator.evaluate()

            self.assertEqual(generated_shots, [0, 1])
            self.assertEqual(
                logged_metrics,
                [
                    {"eval_coco_0shot_f1": 0.25, "eval_coco_0shot_map": 0.125},
                    {"eval_coco_1shot_f1": 1.25, "eval_coco_1shot_map": 1.125},
                ],
            )
            self.assertEqual(
                metrics,
                {
                    "eval_coco_0shot_f1": 0.25,
                    "eval_coco_0shot_map": 0.125,
                    "eval_coco_1shot_f1": 1.25,
                    "eval_coco_1shot_map": 1.125,
                },
            )
            self.assertFalse((root / "generation_eval").exists())

    def test_epoch_boundary_stays_on_the_epoch_that_just_finished(self):
        trainer = _load_trainer_without_optional_training_dependencies()

        label = trainer.format_iter_epoch(
            global_step=370,
            epoch=1.0,
            total_epochs=4.0,
        )

        self.assertEqual(label, "370[1/4]")

    def test_logs_distinguish_micro_batches_from_optimizer_updates(self):
        trainer = _load_trainer_without_optional_training_dependencies()
        configurations = (
            (1, 8, 2958, 80),
            (2, 4, 1479, 40),
        )

        for batch_size, accumulation_steps, micro_batches, completed_micro_batches in configurations:
            with self.subTest(
                batch_size=batch_size,
                accumulation_steps=accumulation_steps,
            ):
                callback = trainer.TrainingTelemetryCallback(swanlab_enabled=False)
                args = types.SimpleNamespace(
                    report_to=[],
                    world_size=4,
                    per_device_train_batch_size=batch_size,
                    gradient_accumulation_steps=accumulation_steps,
                    num_train_epochs=4,
                    dataloader_num_workers=2,
                )
                state = types.SimpleNamespace(
                    is_world_process_zero=True,
                    global_step=10,
                    max_steps=1480,
                    epoch=10 / 370,
                )
                callback.on_train_begin(
                    args,
                    types.SimpleNamespace(global_step=0),
                    control=None,
                    train_dataloader=range(micro_batches),
                )

                with (
                    patch.object(trainer, "_nvidia_smi_metrics", return_value={}),
                    self.assertLogs(trainer.logger, level="INFO") as captured,
                ):
                    callback.on_log(
                        args,
                        state,
                        control=None,
                        logs={"loss": 1.0, "learning_rate": 2e-4},
                    )

                output = "\n".join(captured.output)
                self.assertIn("epoch_update=10/370", output)
                self.assertIn(
                    f"micro_batch={completed_micro_batches}/{micro_batches}",
                    output,
                )

    def test_eval_only_logs_are_printed(self):
        trainer = _load_trainer_without_optional_training_dependencies()
        callback = trainer.TrainingTelemetryCallback(swanlab_enabled=False)
        args = types.SimpleNamespace(report_to=[])
        state = types.SimpleNamespace(is_world_process_zero=True, global_step=3)
        metrics = {
            "eval_coco_1shot_f1": 0.25,
            "eval_coco_1shot_map": 0.125,
        }

        with self.assertLogs(trainer.logger, level="INFO") as captured:
            callback.on_log(args, state, control=None, logs=metrics)

        output = "\n".join(captured.output)
        self.assertIn("eval_coco_1shot_f1=0.2500", output)
        self.assertIn("eval_coco_1shot_map=0.1250", output)

    def test_4x3090_launcher_defaults_to_epoch_coco_eval(self):
        environment = os.environ.copy()
        environment.update(
            {
                "DRY_RUN": "1",
                "DATASET": "data/coco/train_sft_10pct_1to2to4_11829_inst-v5.json",
                "RUN_ID": "telemetry-test",
                "SWANLAB_API_KEY": "test-key",
            }
        )
        environment.pop("REPORT_TO", None)
        result = subprocess.run(
            ["bash", "scripts/train_lora_r64_4x3090.sh"],
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )

        command = result.stdout
        self.assertIn("--report-to swanlab", command)
        self.assertIn("--eval-mode generation", command)
        self.assertIn("--eval-strategy epoch", command)
        self.assertIn("--num-train-epochs 4", command)
        self.assertIn("--save-strategy epoch", command)
        self.assertIn("--save-total-limit 4", command)
        self.assertIn(
            "--coco-eval-episodes data/coco/val_episodes_500_0shot_inst-v5.json "
            "data/coco/val_episodes_500_124_inst-v5.json",
            command,
        )
        self.assertIn("--coco-eval-max-new-tokens 1024", command)

    def test_4x3090_launcher_requires_swanlab_key_before_starting(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            environment = os.environ.copy()
            environment.update(
                {
                    "OUTPUT_DIR": str(output_dir),
                    "PYTHON_BIN": "true",
                    "REPORT_TO": "swanlab",
                }
            )
            environment.pop("SWANLAB_API_KEY", None)

            result = subprocess.run(
                ["bash", "scripts/train_lora_r64_4x3090.sh"],
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("SWANLAB_API_KEY must be set", result.stderr)
            self.assertFalse(output_dir.exists())

    def test_4x3090_launcher_writes_a_training_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "captured-args"
            interpreter = root / "capture-python"
            interpreter.write_text(
                "#!/bin/sh\n"
                "if [ \"$1\" = \"-\" ]; then\n"
                "  cat >/dev/null\n"
                "  exit 0\n"
                "fi\n"
                "printf '%s\\n' \"$@\" > \"$CAPTURE_ARGS\"\n"
            )
            interpreter.chmod(0o755)
            environment = os.environ.copy()
            environment.update(
                {
                    "PYTHON_BIN": str(interpreter),
                    "CAPTURE_ARGS": str(capture),
                    "OUTPUT_DIR": str(root / "run"),
                    "REPORT_TO": "none",
                }
            )

            subprocess.run(
                ["bash", "scripts/train_lora_r64_4x3090.sh"],
                cwd=ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )

            log_file = root / "run" / "train.log"
            self.assertTrue(log_file.is_file())
            self.assertIn(f"log_file={log_file}", log_file.read_text())


if __name__ == "__main__":
    unittest.main()
