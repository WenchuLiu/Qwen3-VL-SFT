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
    generation.evaluate_loaded_model = None
    generation.load_episodes = None
    generation.result_payload = None
    metrics = types.ModuleType("qwen3vl_sft.evaluation.coco.metrics")
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
            "--coco-eval-episodes data/coco/val_episodes_500_124_inst-v5.json",
            command,
        )

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
