# Project structure and entry-point policy

This repository has three different kinds of evaluation. They should not be
implemented as three unrelated copies of an evaluator:

| Layer | Location | Purpose |
| --- | --- | --- |
| Training validation | `qwen3vl_sft/train/` | `--eval-mode loss` or `generation` during training |
| Fixed COCO evaluation | `qwen3vl_sft/evaluation/coco/` | One canonical prompt, generation, parsing, and mAP implementation |
| Cross-domain few-shot benchmark | `qwen3vl_sft/evaluation/fewshot/` | Dataset registry, episode manifests, cache, and multi-GPU scheduling |

Visual Enhancement is a mode of the shared generation protocol. It draws boxes
on support images through the common prompt builder; it is not a fourth copy of
the evaluator.

## Canonical tree

```text
qwen3-vl-sft/
├── qwen3vl_sft/                 # importable source of truth
│   ├── model/                   # model loading and LoRA policy
│   ├── data/                    # schema, preprocessing, RoPE, collators
│   ├── train/                   # SFT, generation validation, and GRPO
│   └── evaluation/
│       ├── coco/                # canonical COCO protocol and metrics
│       └── fewshot/             # cross-domain benchmark application
├── tools/                       # thin Python CLI entry points
├── scripts/                     # canonical operational shell entry points
│   ├── train_lora.sh
│   ├── train_grpo.sh
│   ├── build_coco_train.sh
│   ├── build_coco_eval.sh
│   ├── evaluate_coco.sh
│   ├── evaluate_fewshot.sh      # baseline and VE via VE=1
│   └── run_coco_before_after.sh
├── docs/                        # active project and protocol documentation
├── document/                    # older detailed documents kept for links
├── shell/                       # compatibility wrappers only
├── scripts/cross_domain_datasets/
│                                # compatibility dataset/VE launcher paths
├── data/                        # local datasets/manifests; git-ignored
├── weights/                     # local checkpoints; git-ignored
└── outputs/                     # generated training/evaluation artifacts; git-ignored
    ├── train/                   # SFT/GRPO model outputs, checkpoints, and logs
    ├── eval/                    # standalone evaluation results and manifests
    └── experiments/             # multi-stage before/after experiment bundles
```

## Which command should be used?

Use the package entry point for new training jobs:

```bash
torchrun --nproc_per_node=2 -m qwen3vl_sft.train ...
```

Use the canonical scripts for reproducible operations:

```bash
bash scripts/train_lora.sh
bash scripts/train_grpo.sh
bash scripts/evaluate_coco.sh
bash scripts/evaluate_fewshot.sh
VE=1 bash scripts/evaluate_fewshot.sh
```

`tools/*.py` remains useful for direct CLI access and backward compatibility.
The old `shell/*.sh` and `scripts/cross_domain_datasets/*.sh` paths are wrappers
or dataset-specific compatibility launchers; new experiments should not add
more logic there.

## Path policy

No experiment path should contain a developer's home directory. Shell scripts
may resolve their own repository root at runtime with `dirname` and `pwd`; this
is an implementation detail, not a hard-coded machine path. Defaults should be
repository-relative (`data`, `weights`, `outputs`). External datasets
and checkpoints are supplied through environment variables such as
`DATA_ROOT`, `COCO_ROOT`, `MODEL_PATH`, or `MODEL_NAME_OR_PATH`.

The tracked source tree must not contain generated checkpoints, manifests,
logs, caches, or machine-specific Slurm paths. Those belong under the ignored
runtime directories or in a user-provided external data root.

## Result ownership

- `data/`: input datasets and deterministic manifests.
- `outputs/train/`: one directory per SFT or GRPO run. The run directory is
  passed to Hugging Face Trainer as `output_dir` and contains its checkpoints,
  final model/adapter, trainer state, arguments, and local telemetry.
- `outputs/eval/`: standalone COCO and cross-domain few-shot manifests,
  predictions, summaries, and evaluation logs.
- `outputs/experiments/`: bundled workflows that combine data preparation,
  training, base/adapter evaluation, and comparison results.

The legacy `runs/` and `work_dirs/` directories remain ignored and can still be
used through explicit `OUTPUT_DIR`/`WORK_ROOT` overrides, but new scripts should
write under `outputs/`.

Keep base, adapter, baseline, and VE outputs in separate subdirectories. This
makes `--skip-existing` safe and prevents comparing results produced by
different prompt or preprocessing modes.
