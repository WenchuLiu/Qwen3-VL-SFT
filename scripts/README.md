# Canonical scripts

This directory contains the repository's supported operational entry points.
Run them from any working directory; each script resolves the repository root
from its own location and uses repository-relative defaults.

| Script | Role |
| --- | --- |
| `train_lora.sh` | supervised LoRA/full-parameter training |
| `train_grpo.sh` | score-aware multimodal GRPO |
| `build_coco_train.sh` | build COCO SFT records |
| `build_coco_eval.sh` | build a fixed COCO episode manifest |
| `evaluate_coco.sh` | evaluate one fixed COCO manifest |
| `evaluate_coco_benchmark.sh` | build (if needed) and evaluate the COCO manifest |
| `evaluate_fewshot.sh` | evaluate cross-domain few-shot datasets |
| `run_coco_before_after.sh` | build, evaluate base, train, evaluate adapter |

Visual Enhancement uses the same few-shot evaluator:

```bash
VE=1 bash scripts/evaluate_fewshot.sh
```

Instruction Enhancement uses a category-description JSON mapping:

```bash
IE=1 CATEGORY_DESCRIPTIONS=docs/category_descriptions.example.json \
bash scripts/evaluate_fewshot.sh
```

DetPO uses the detailed, dataset- and shot-specific prompts under
`docs/cross-domain-instructions`:

```bash
DETPO=1 bash scripts/evaluate_fewshot.sh
```

The existing DetPO launcher defaults to 1/2/4-shot episodes and selects
`{shot}-shot/all_refined_class_instructions_{dataset}.json` for each task.
It uses the original DetPO single-image prompt rather than the IE ICL
question; the shot controls which optimized description file is selected.
Override the prompt root with `DETPO_PROMPTS=/path/to/prompts`.

The repository includes the complete 57-category mapping for ArTaxOr,
Clipart1k, FISH, NEU-DET, UODD, and VISUALDIOR at
`docs/cross_domain_category_descriptions.json`.

Without `MAX_NEW_TOKENS`, the evaluator uses the dataset registry: VISUALDIOR
gets 2,048 new tokens and all other registered cross-domain datasets get 1,024.
Set `MAX_NEW_TOKENS` only when a deliberate global override is needed.

Override external resources without editing scripts:

```bash
MODEL_PATH=/path/to/checkpoint \
DATA_ROOT=/path/to/benchmark-data \
WORK_ROOT=outputs/eval/fewshot/my-run \
bash scripts/evaluate_fewshot.sh
```

Training defaults are under `outputs/train/`, while standalone evaluation
defaults are under `outputs/eval/`. Set `OUTPUT_DIR`, `WORK_ROOT`, or `EVAL_ROOT`
to place a run on another filesystem without changing the launchers.

The `shell/` directory and `scripts/cross_domain_datasets/` are retained as
compatibility paths. They should delegate to these scripts rather than gain new
logic.
