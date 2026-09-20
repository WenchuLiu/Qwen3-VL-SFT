# Qwen3-VL-SFT

A focused repository for supervised fine-tuning Qwen3-VL with Hugging Face
Transformers. It keeps the official Qwen conversation format and multimodal
RoPE preprocessing, while making the training and evaluation contract explicit.

本仓库专门解决两个容易混淆的问题：

1. `loss` eval 是 teacher-forcing 的语言模型 loss；它不是检测指标。
2. COCO ICL eval 是固定 episodes 上的真实 `generate()`，再解析 bbox 和模型
   自报置信度，计算 `F1@Mean` 与 DetPO 风格 COCO mAP。训练中 eval 和独立
   eval 使用完全相同的协议。

## Design

The repository follows the same separation used by the reference
LocateAnything project: model code, training code, evaluation applications,
and operational launchers are separate package boundaries.

```text
qwen3vl_sft/
  model/         Qwen3-VL loading, LoRA, and trainable-module policies
  data/          schema, messages, single-record preprocessing, and RoPE
  train/         arguments, datasets, collator, Trainer wiring, and runner
  evaluation/    coco/ protocol plus fewshot/ evaluation applications
tools/           thin Python command-line entry points
scripts/         canonical reproducible shell entry points
shell/           compatibility wrappers that delegate to scripts/
scripts/cross_domain_datasets/
                 compatibility paths for old dataset-specific launchers
docs/            active project structure and migration documentation
document/        detailed legacy documents retained for existing links
```

The complete ownership and entry-point policy is documented in
[`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md). In particular,
`VE` is a mode of the shared generation evaluator, not a separate evaluation
implementation.

The source of truth for the COCO protocol is
`qwen3vl_sft/evaluation/coco/protocol.py`; the older flat module path is a
compatibility facade. Do not copy its prompt into a shell script or a second
evaluator. The baseline prompt template remains `inst-v5`; IE is an independent
`detpo-ie-v2` runtime variant that follows DetPO's prompt layout by placing the
category description in a separate annotator-instructions block on the final query. SFT targets omit
confidence, while the final evaluation query asks the model to estimate
confidence for each predicted box.

The main training command is now the package entry point:

```bash
torchrun --nproc_per_node=2 -m qwen3vl_sft.train ...
```

Existing `tools/train.py`, `tools/evaluate_fewshot.py`, and flat
`qwen3vl_sft.*` imports remain supported as thin compatibility launchers.

## Install

For a CUDA-enabled NVIDIA environment, enter the allocated compute node and run
the installer from the repository root:

```bash
salloc -p gpu -N1 -n3 --gres=gpu:1
ssh <allocated-gpu-node>
# cd Qwen3-VL-SFT
bash install_llm_env.sh
conda activate LLM
```

The script creates (or updates) a Python 3.10 Conda environment named `LLM`,
installs PyTorch 2.6 with the default CUDA 11.8 wheel and the repository's
pinned runtime dependencies. It enumerates all visible GPUs and runs a CUDA
matrix-multiplication check when CUDA is available. The installer has no A40
or `sm_86` requirement, and does not install FlashAttention: the project uses
PyTorch `sdpa` by default. Set `TORCH_INDEX_URL` to another official PyTorch
CUDA wheel index when the host driver/GPU needs a different CUDA line. After
leaving the GPU node, release the allocation with `scancel JOBID`.

The complete no-FlashAttention procedure, including a manual virtualenv path,
is documented in [`INSTALL.md`](INSTALL.md).

Alternatively, use a CUDA-compatible PyTorch build and install the package in
a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
```

For multi-GPU training, use a CUDA/PyTorch combination supported by the target
GPU, Transformers, Accelerate, and PEFT. Do not pass `flash_attention_2`; use
`sdpa` (the default) or `eager`. The default image budget is 3,136 to
640,000 pixels (an 800x800-equivalent maximum) for both training and COCO
generation evaluation; override it explicitly in both commands when changing
the budget.

## Dataset format

The training input follows the official Qwen VL format. A record can be a JSON
object in a JSON list, or one object per line in JSONL:

```json
{
  "image": "images/001.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nWhat is in this image?"},
    {"from": "gpt", "value": "A red apple."}
  ]
}
```

Supported roles are `system`, `human`/`user`, and `gpt`/`assistant`. Every
`<image>` and `<video>` placeholder must consume exactly one corresponding
media path. Answers are the only default supervised spans. For a multi-turn
ICL record, set `"loss_mode": "last_assistant"` to train only the final query
answer and keep support answers as context.

## Train

The most direct entry point is:

```bash
torchrun --nproc_per_node=2 -m qwen3vl_sft.train \
  --model-name-or-path Qwen/Qwen3-VL-4B-Instruct \
  --dataset data/my_train.json \
  --output-dir outputs/train/sft/my-lora \
  --lora-enable true \
  --bf16 true \
  --eval-mode none
```

The equivalent wrapper is `scripts/train_lora.sh`:

```bash
export SWANLAB_API_KEY=your_api_key
MODEL_NAME_OR_PATH=Qwen/Qwen3-VL-4B-Instruct \
DATASET=data/my_train.json \
OUTPUT_DIR=outputs/train/sft/my-lora \
bash scripts/train_lora.sh
```

The wrapper reports to SwanLab by default using project
`qwen3vl-coco-sft`, names the run after `OUTPUT_DIR`, and trains for 12 epochs.
Override these defaults with `SWANLAB_PROJECT`, `RUN_NAME`, `REPORT_TO`, or
`NUM_TRAIN_EPOCHS`. Keep `SWANLAB_API_KEY` in the environment; do not store it
in a script or configuration committed to Git.

By default, new training runs are created under
`outputs/train/sft/<run-id>/` or `outputs/train/grpo/<run-id>/`. Set
`RUN_ID` for a stable, human-readable run name or set `OUTPUT_DIR` to an
explicit path when resuming an existing run.

## Score-aware GRPO fine-tuning

The repository also includes a standalone multimodal GRPO path modeled after
`../Visual-RFT`. It samples several answers for each prompt, computes a
group-relative reward, and updates the policy with a KL penalty. The default
reward is the sum of:

- one-to-one box IoU reward;
- score reward: high score for an IoU-matched box and low score for an
  unmatched box;
- a format reward requiring a parseable detection list with explicit
  `score` fields.

Existing COCO `train_sft.json` records can be used directly. The GRPO loader
removes the final supervised answer, changes the final query to request
confidence scores, and keeps the answer's boxes as reward targets.

```bash
MODEL_NAME_OR_PATH=Qwen/Qwen3-VL-4B-Instruct \
DATASET=data/coco/train_sft.json \
OUTPUT_DIR=outputs/train/grpo/coco-grpo \
NPROC_PER_NODE=2 \
bash scripts/train_grpo.sh
```

The direct entry point is `python -m qwen3vl_sft.train.grpo`. Useful controls
include `--num-generations`, `--max-completion-length`, `--kl-coef`,
`--iou-threshold`, and `--reward-functions iou score format`. LoRA is enabled
by default, so the initial policy can be used as the KL reference by disabling
the adapter; full-parameter GRPO creates a frozen reference copy automatically
or can load one explicitly with `--reference-model-name-or-path`.

For ordinary held-out language-model validation:

```bash
torchrun --nproc_per_node=2 -m qwen3vl_sft.train \
  --model-name-or-path Qwen/Qwen3-VL-4B-Instruct \
  --dataset data/train.json \
  --eval-mode loss \
  --eval-ratio 0.05 \
  --eval-strategy epoch \
  --output-dir outputs/train/sft/loss-eval
```

For LoRA, the default target modules are the language-model
`q_proj/k_proj/v_proj/o_proj`. LoRA freezes the vision tower. Full or selected
module training is available with `--lora-enable false` and the
`--tune-mm-*` options.

## COCO ICL detection experiment

Build the train records and the fixed validation episodes separately. The
episode file is an immutable evaluation manifest: changing the seed, split,
support pool, or shot list means creating a new manifest.

```bash
export COCO_ROOT=data/coco
bash scripts/build_coco_train.sh
bash scripts/build_coco_eval.sh
```

The training-data wrapper defaults to the deterministic 10% train2017 image
pool (11,829 selected image IDs), 11,829 category-balanced episodes, and a
near-uniform 1/2/4-shot mixture (3-shot is intentionally excluded).

Evaluate the base model:

```bash
MODEL_NAME_OR_PATH=Qwen/Qwen3-VL-4B-Instruct \
EPISODES=data/coco/val_episodes.json \
OUTPUT=outputs/eval/coco/base/result.json \
bash scripts/evaluate_coco.sh
```

For the standard 500-query-image evaluation, the standalone COCO launcher
builds `data/coco/val_episodes.json` when needed and uses `1024` new tokens:

```bash
COCO_ROOT=data/coco \
MODEL_NAME_OR_PATH=Qwen/Qwen3-VL-4B-Instruct \
bash scripts/evaluate_coco_benchmark.sh
```

Train and evaluate with generation-based validation:

```bash
MODEL_NAME_OR_PATH=Qwen/Qwen3-VL-4B-Instruct \
DATASET=data/coco/train_sft.json \
OUTPUT_DIR=outputs/train/sft/coco-lora \
EVAL_MODE=generation \
EVAL_STRATEGY=epoch \
EVAL_EPISODES=data/coco/val_episodes.json \
bash scripts/train_lora.sh
```

Evaluate the adapter using the same fixed manifest:

```bash
MODEL_NAME_OR_PATH=Qwen/Qwen3-VL-4B-Instruct \
ADAPTER_PATH=outputs/train/sft/coco-lora \
EPISODES=data/coco/val_episodes.json \
OUTPUT=outputs/eval/coco/adapter/result.json \
bash scripts/evaluate_coco.sh
```

## Local few-shot benchmark

The few-shot benchmark has one canonical launcher. It evaluates the original
local 4B checkpoint on ArTaxOr, Clipart1k, FISH, NEU-DET, UODD, and VISUALDIOR
at 0/1/2/4-shot. It uses each dataset's fixed
`annotations/{1,2,4}_shot.json` support file and evaluates every positive test
image/category pair with official COCO mAP. Its generation budgets match the
configured cross-domain evaluation: `1024` new tokens for ArTaxOr,
Clipart1k, FISH, NEU-DET, and UODD, and `2048` for VISUALDIOR. Set
`MAX_NEW_TOKENS` only when a global override is intended:

```bash
MODEL_PATH=weights/Qwen3-VL-4B-Instruct \
DATA_ROOT=data \
bash scripts/evaluate_fewshot.sh
```

For example, `MAX_NEW_TOKENS=2048` overrides the dataset-specific default for
all selected datasets. Select a subset with `DATASETS` and shots with `SHOTS`:

```bash
DATASETS="FISH VISUALDIOR" SHOTS="1 2 4" \
bash scripts/evaluate_fewshot.sh
```

Individual runs write to `outputs/eval/fewshot/qwen3-vl-4b-fewshot/<dataset>/`, keeping
each dataset's `config.json`, `summary.json`, results, and `evaluation.log`
separate.

The baseline launcher defaults to two persistent GPU workers (`cuda:0` and `cuda:1`),
each with its own 4B model replica. Override with `NUM_GPUS=1` when only one
GPU is available.

Visual Enhancement is available as an optional evaluation mode. It draws the
ground-truth boxes on support images only; query images are never annotated.
Run the launcher from the repository root; its default model, data, and output
paths are project-relative.
The canonical launcher evaluates ArTaxOr, Clipart1k, FISH, NEU-DET, UODD, and
VISUALDIOR at 1/2/4 shots with four GPU workers, batch size 2, an 800x800
maximum image budget, and dataset-specific generation budgets (1,024 for all
datasets except 2,048 for VISUALDIOR):

```bash
VE=1 \
MODEL_PATH=weights/Qwen3-VL-4B-Instruct \
DATA_ROOT=data \
NUM_GPUS=4 \
bash scripts/evaluate_fewshot.sh
```

The same mode can be selected for one dataset with
`VE=1 DATASETS=FISH SHOTS="1 2 4" bash scripts/evaluate_fewshot.sh`. The old
`scripts/cross_domain_datasets/run_ve.sh` path remains a compatibility wrapper. VE
requires at least one support shot, so it cannot be combined with `SHOTS="0"`.

Instruction Enhancement is a separate training-free prompt variant. It adds a
user-provided visual description of the requested category to the final query
as a DetPO-style annotator-instructions block; support demonstrations retain the
baseline question format. It does not change images, weights, or the evaluation
metric. The description file is a JSON object mapping category names to English
descriptions, optionally wrapped under `descriptions` (see
`docs/category_descriptions.example.json`):

```bash
IE=1 \
CATEGORY_DESCRIPTIONS=docs/category_descriptions.example.json \
DATASETS=FISH SHOTS="0 1 2 4" \
MODEL_PATH=weights/Qwen3-VL-4B-Instruct \
DATA_ROOT=data NUM_GPUS=1 \
bash scripts/evaluate_fewshot.sh
```

Every category appearing in the selected episodes must have a description.
IE also works with `VE=1` to test the combined visual-and-instruction variant;
the result records `instruction_enhancement` and the description-file hash so
cached baseline, VE, and IE runs remain distinguishable.
The compatibility wrapper `scripts/cross_domain_datasets/run_ie.sh` sets
`IE=1` for existing launcher workflows.

For all six registered cross-domain datasets, use the complete 57-category
mapping generated for this benchmark:

```bash
IE=1 \
CATEGORY_DESCRIPTIONS=docs/cross_domain_category_descriptions.json \
MODEL_PATH=weights/Qwen3-VL-4B-Instruct DATA_ROOT=data NUM_GPUS=4 \
bash scripts/evaluate_fewshot.sh
```

Results follow an MMDetection-style layout under the selected `WORK_ROOT`
(default `outputs/eval/fewshot/qwen3-vl-4b-ve-fewshot/` for VE): `evaluation.log`, `config.json`,
`summary.json`, and one `episodes.json` plus `result.json` per dataset/shot.
Use `SKIP_EXISTING=1` to resume completed entries. The main metric is
`map_50_95`; `map_50`, `map_75`, ranking-mAP, raw responses, and parsed
predictions are retained in each result file.

The complete base-before/adapter-after workflow is available as:

```bash
COCO_ROOT=data/coco \
MODEL_NAME_OR_PATH=Qwen/Qwen3-VL-4B-Instruct \
bash scripts/run_coco_before_after.sh
```

## What counts as eval?

| Mode | Model call | Target | Metric | Intended use |
| --- | --- | --- | --- | --- |
| `none` | no eval | none | none | fastest training |
| `loss` | forward pass | held-out assistant tokens | `eval_loss` | language-model fit |
| `generation` | greedy `generate()` | fixed episode query boxes | `eval_coco_*` F1 + mAP | detection quality |

`generation` writes `generation_eval/step-*.json` under the training output
directory. Standalone evaluation writes the same result schema. This avoids
selecting a checkpoint using one protocol and reporting it with another.

## Reproducibility rules

- Build the train data and validation episode manifest once, then reuse them.
- Keep support images from the train split and query images from the validation
  split.
- Use the same pixel budgets, max new tokens, greedy decoding, prompt version,
  and episode file before and after SFT.
- Never evaluate a checkpoint by sampling a new support set implicitly.
- Do not commit model weights, COCO data, or experiment outputs; `.gitignore`
  excludes them.

## Upstream basis

The data schema, model-family handling, multimodal position IDs, and LoRA
target conventions are based on the official Qwen3-VL repository's
`qwen-vl-finetune` implementation, particularly the `QwenLM/Qwen3-VL` `main`
revision available when this repository was prepared. The implementation is
organized here as an independent project so experiments do not depend on the
full Qwen demo and benchmark tree.

Useful references:

- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- [Qwen2.5-VL fine-tuning directory](https://github.com/QwenLM/Qwen2.5-VL/tree/main/qwen-vl-finetune)
- [Transformers Qwen3-VL documentation](https://huggingface.co/docs/transformers)

## License

This project is released under Apache License 2.0. Some preprocessing ideas
are adapted from the official Qwen fine-tuning example; retain the upstream
copyright and license notices when redistributing derived code.
