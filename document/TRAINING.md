# Training

## Install

```bash
python -m pip install -e '.[dev]'
```

GPU training requires a CUDA-compatible PyTorch installation. The repository's
`install_llm_env.sh` installs the pinned environment without requiring a
specific GPU model; see `INSTALL.md` for CUDA wheel overrides.

## Entry points

Use the package entry point for new runs:

```bash
torchrun --nproc_per_node=2 -m qwen3vl_sft.train \
  --model-name-or-path Qwen/Qwen3-VL-4B-Instruct \
  --dataset data/train.json \
  --output-dir runs/qwen3vl-lora \
  --lora-enable true \
  --bf16 true \
  --eval-mode none
```

The `scripts/train_lora.sh` and `shell/train_lora.sh` launchers expose the
same configuration through environment variables. `tools/train.py` is kept as
a compatibility launcher for older jobs.

## Data contract

Training accepts a JSON list/object or JSONL file containing the official Qwen
conversation schema. Images and videos are resolved relative to the annotation
file unless `--data-root` or a record-level `data_path` is provided. Each media
placeholder must consume exactly one media path.

The dataset layer and collator are separate from single-record processing:

- `train/dataset.py` indexes files and makes deterministic train/eval splits.
- `data/processing.py` applies the chat template and creates assistant labels.
- `train/collator.py` pads text and concatenates vision tensors.

## Evaluation modes

`none` disables evaluation, `loss` evaluates held-out assistant tokens with a
forward pass, and `generation` evaluates fixed COCO episodes with
`generate()`. Generation mode requires `--coco-eval-episodes` and an eval
schedule.

LoRA is the default. Disable it with `--lora-enable false` and explicitly
select the modules to tune with `--tune-mm-vision`, `--tune-mm-mlp`, and
`--tune-mm-llm`.

## Score-aware GRPO

Use the standalone GRPO entry point for reinforcement learning:

```bash
MODEL_NAME_OR_PATH=Qwen/Qwen3-VL-4B-Instruct \
DATASET=data/coco/train_sft.json \
OUTPUT_DIR=runs/qwen3vl-grpo \
bash scripts/train_grpo.sh
```

`qwen3vl_sft.train.grpo_data` accepts the repository's SFT conversation
records, COCO episode records with `support`/`query`, and generic records with
a conversational `prompt`. The final answer is not fed to the model during a
rollout. Its boxes are attached as reward metadata, while the final prompt is
rewritten to request `bbox_2d`, `label`, and a confidence `score` in `[0, 1]`.

The default registry is `iou score format`. IoU uses greedy one-to-one
matching at `--iou-threshold`; the score reward gives matched detections their
emitted score and unmatched detections `1 - score`. Missing score fields get
zero score reward. Group advantages are normalized over
`--num-generations` samples and optimized with the sampled-token policy loss
plus `--kl-coef` KL regularization.

This first GRPO path intentionally has no generation-based validation loop;
run the existing fixed-episode COCO evaluator after each saved checkpoint.
