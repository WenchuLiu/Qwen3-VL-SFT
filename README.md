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

The repository has four responsibilities:

```text
qwen3vl_sft/
  data/          Qwen conversation parsing, labels, multimodal collation, RoPE
  evaluation/    COCO episode protocol, generation, parsing, and metrics
  modeling.py    model loading, LoRA, and trainable-module selection
  train.py       one training entry point
tools/           command-line entry points
scripts/         reproducible shell wrappers
tests/           CPU-only protocol and metric tests
```

The source of truth for the COCO protocol is
`qwen3vl_sft/evaluation/coco_protocol.py`. Do not copy its prompt into a shell
script or a second evaluator. The default and only supported prompt template is
`inst-v4`: SFT targets omit confidence, while the final evaluation query asks
the model to estimate confidence for each predicted box.

## Install

For the pinned NVIDIA A40 training environment, first request one GPU and enter
the node allocated by Slurm. For example, if the allocation reports `gpu1`:

```bash
salloc -p gpu -N1 -n3 --gres=gpu:1
ssh gpu1
cd /home/u1120240334/code/Qwen3-VL-SFT
bash install_llm_env.sh
conda activate LLM
```

The script creates (or updates) a Python 3.10 Conda environment named `LLM`,
installs PyTorch 2.6 with CUDA 11.8 and an environment-local CUDA toolkit, then
builds FlashAttention for the A40's `sm_86` architecture. It finishes with an
actual BF16 CUDA operation. CUDA 11.8 is intentional: it supports the A40 while
requiring an older cluster driver than CUDA 12.4. After leaving the GPU node,
release the allocation with `scancel JOBID`.

Alternatively, use a CUDA-compatible PyTorch build and install the package in
a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
```

For multi-GPU training, use a recent CUDA/PyTorch combination supported by
Transformers, Accelerate, and PEFT. `flash-attn` is optional; the default
attention implementation is `sdpa`. The default image budget is 3,136 to
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
  --dataset /data/my_train.json \
  --output-dir runs/my-lora \
  --lora-enable true \
  --bf16 true \
  --eval-mode none
```

The equivalent wrapper is `scripts/train_lora.sh`:

```bash
export SWANLAB_API_KEY=your_api_key
MODEL_NAME_OR_PATH=Qwen/Qwen3-VL-4B-Instruct \
DATASET=/data/my_train.json \
OUTPUT_DIR=runs/my-lora \
bash scripts/train_lora.sh
```

The wrapper reports to SwanLab by default using project
`qwen3vl-coco-sft`, names the run after `OUTPUT_DIR`, and trains for 12 epochs.
Override these defaults with `SWANLAB_PROJECT`, `RUN_NAME`, `REPORT_TO`, or
`NUM_TRAIN_EPOCHS`. Keep `SWANLAB_API_KEY` in the environment; do not store it
in a script or configuration committed to Git.

For ordinary held-out language-model validation:

```bash
torchrun --nproc_per_node=2 -m qwen3vl_sft.train \
  --model-name-or-path Qwen/Qwen3-VL-4B-Instruct \
  --dataset /data/train.json \
  --eval-mode loss \
  --eval-ratio 0.05 \
  --eval-strategy epoch \
  --output-dir runs/loss-eval
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
export COCO_ROOT=/data/coco
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
OUTPUT=runs/base.json \
bash scripts/evaluate_coco.sh
```

Train and evaluate with generation-based validation:

```bash
MODEL_NAME_OR_PATH=Qwen/Qwen3-VL-4B-Instruct \
DATASET=data/coco/train_sft.json \
OUTPUT_DIR=runs/coco-lora \
EVAL_MODE=generation \
EVAL_STRATEGY=epoch \
EVAL_EPISODES=data/coco/val_episodes.json \
bash scripts/train_lora.sh
```

Evaluate the adapter using the same fixed manifest:

```bash
MODEL_NAME_OR_PATH=Qwen/Qwen3-VL-4B-Instruct \
ADAPTER_PATH=runs/coco-lora \
EPISODES=data/coco/val_episodes.json \
OUTPUT=runs/adapter.json \
bash scripts/evaluate_coco.sh
```

The complete base-before/adapter-after workflow is available as:

```bash
COCO_ROOT=/data/coco \
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
