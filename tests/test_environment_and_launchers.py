import os
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class EnvironmentInstallerTest(unittest.TestCase):
    def test_installer_verifies_coco_runtime_imports(self):
        installer = (ROOT / "install_llm_env.sh").read_text()

        self.assertIn("from pycocotools.coco import COCO", installer)
        self.assertIn("from pycocotools.cocoeval import COCOeval", installer)
        self.assertIn('version("pycocotools")', installer)


class TwoGpuLauncherTest(unittest.TestCase):
    def test_2x3090_launcher_preserves_effective_batch_and_eval_defaults(self):
        environment = os.environ.copy()
        environment.update(
            {
                "DRY_RUN": "1",
                "REPORT_TO": "none",
                "RUN_ID": "two-gpu-launcher-test",
            }
        )
        result = subprocess.run(
            ["bash", "scripts/train_lora_r64_2x3090.sh"],
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )

        command = result.stdout
        self.assertIn("--nproc_per_node=2", command)
        self.assertIn("--per-device-train-batch-size 2", command)
        self.assertIn("--gradient-accumulation-steps 4", command)
        self.assertIn("--eval-mode generation", command)
        self.assertIn("--coco-eval-max-new-tokens 1024", command)

    def test_2x3090_launcher_defaults_to_two_visible_gpus(self):
        launcher = (ROOT / "scripts/train_lora_r64_2x3090.sh").read_text()

        self.assertIn('CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"', launcher)


if __name__ == "__main__":
    unittest.main()
