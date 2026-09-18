# Migration From The Mixed Repository

| Old location | New location |
| --- | --- |
| `qwen-vl-finetune/qwenvl/train/train_qwen.py` | `qwen3vl_sft/train/runner.py` |
| `qwen-vl-finetune/qwenvl/data/data_processor.py` | `qwen3vl_sft/data/processing.py` |
| `incontext_det/build_coco_incontext_sft.py` | `tools/build_coco.py train` |
| `incontext_det/build_coco_incontext_eval.py` | `tools/build_coco.py eval` |
| `incontext_det/evaluate_coco_incontext.py` | `tools/evaluate_coco.py` |
| `exp_baseline.sh` | `scripts/evaluate_coco.sh` |
| `run_coco_icl_before_after_sft.sh` | `scripts/run_coco_before_after.sh` |

For new training jobs, use `python -m qwen3vl_sft.train` (or `torchrun -m
qwen3vl_sft.train`). The former flat Python modules and `tools/` launchers are
kept as compatibility facades, so existing experiment commands do not need to
be migrated in one step.

The new repository intentionally does not copy unrelated web demos, benchmark
launchers, attention interventions, generated results, caches, or model files.
Those belong in separate research repositories or experiment artifacts.
