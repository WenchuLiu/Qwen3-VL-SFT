# Training

## Install

```bash
python -m pip install -e '.[dev]'
```

GPU training requires a CUDA-compatible PyTorch installation. The repository's
`install_llm_env.sh` remains the cluster-specific A40 setup recipe.

## Entry points

Use the package entry point for new runs:

```bash
torchrun --nproc_per_node=2 -m qwen3vl_sft.train \
  --model-name-or-path Qwen/Qwen3-VL-4B-Instruct \
  --dataset /data/train.json \
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
