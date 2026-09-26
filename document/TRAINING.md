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
  --output-dir outputs/train/sft/qwen3vl-lora \
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

## Four RTX 3090 GPUs: r64 LoRA SFT

`scripts/train_lora_r64_4x3090.sh` is a single-node DDP preset for the local
Qwen3-VL-4B-Instruct checkpoint. It uses BF16, SDPA, non-reentrant gradient
checkpointing, rank 64 / alpha 128, and the existing attention-only LoRA
targets. The effective batch is 4 GPUs x 2 samples x 2 accumulation steps = 16.
Learning rate is 5e-5 with 3% warmup and cosine decay; the initial run is three
epochs. These are starting settings, not measured optimums or a memory guarantee.

Activate the CUDA training environment first, then run from the repository:

```bash
DATASET=data/coco/train_sft_10pct_1to4_11829.json \
DRY_RUN=1 bash scripts/train_lora_r64_4x3090.sh

CUDA_VISIBLE_DEVICES=0,1,2,3 \
DATASET=data/coco/train_sft_10pct_1to4_11829.json \
bash scripts/train_lora_r64_4x3090.sh
```

`DRY_RUN=1` only prints the command. Actual launch checks that four CUDA GPUs
are visible. Use `PYTHON_BIN` to select an interpreter. Paths are relative to
the repository root; `DATA_ROOT` defaults to that root because the local COCO
manifests contain paths such as `data/COCO/train2017/...`. Override `DATA_ROOT`
for datasets using another convention.

The preset preserves the other experiments' 800 x 800 pixel budget: each image
is limited to 640000 pixels (approximately 625 visual tokens
with Qwen3-VL's 16-pixel patches and 2x spatial merge) and each full conversation
to 4096 tokens. Multiple support images count separately. Long answers or many
shots can still exceed the sequence budget: the collator raises an error rather
than truncating vision tokens or silently dropping supervision. Inspect your
dataset lengths before a long run. Keep this pixel budget unchanged for
comparability with the other experiments; handle memory pressure through batch
size, checkpointing, or model/runtime memory optimizations instead.
Four DDP workers each hold a complete model; their VRAM is not pooled.

For two RTX 3090 GPUs, use the matching preset:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
DATASET=data/coco/train_sft_10pct_1to4_11829.json \
bash scripts/train_lora_r64_2x3090.sh
```

It uses two DDP workers and defaults to 4 gradient accumulation steps, keeping
the same effective batch size of 16 as the four-GPU preset. All other training,
logging, checkpoint, and 0/1/2/4-shot evaluation defaults are shared.

Environment variables override the preset values shown in the script. Additional
CLI arguments are forwarded, so a short training check is:

```bash
DATASET=data/coco/train_sft_10pct_1to4_11829.json \
bash scripts/train_lora_r64_4x3090.sh \
  --max-steps 5 --logging-steps 1 --save-strategy no
```

Training still saves the final adapter and processor with `--save-strategy no`.
Runs get a timestamped output directory and do not automatically resume. To
continue an interrupted run, set both `OUTPUT_DIR` and `RESUME_FROM_CHECKPOINT`.
Logging defaults to `REPORT_TO=none`; use `REPORT_TO=swanlab` with the SDK's
normal authentication and `SWANLAB_PROJ_NAME` for its project name.

The preset evaluates fixed held-out episodes after every epoch. It runs
0/1/2/4-shot tasks in order and shards each task across all DDP ranks using the
same strided split as the standalone few-shot evaluator. Rank zero restores the
original episode order, computes F1 and mAP, and prints the metrics to the
training log; online evaluation does not create a separate result directory.
The script permits other model paths, but the 8B model needs its own memory
measurement before reusing these settings. This preset is for image SFT, not a
tested video-training configuration.

## Score-aware GRPO

Use the standalone GRPO entry point for reinforcement learning:

```bash
MODEL_NAME_OR_PATH=Qwen/Qwen3-VL-4B-Instruct \
DATASET=data/coco/train_sft.json \
OUTPUT_DIR=outputs/train/grpo/qwen3vl-grpo \
bash scripts/train_grpo.sh
```

To continue GRPO from a LoRA SFT run, pass the base model and adapter
separately. The adapter is loaded as trainable and GRPO updates those existing
LoRA weights:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
MODEL_NAME_OR_PATH=weights/Qwen3-VL-4B-Instruct \
ADAPTER_PATH=outputs/train/sft/qwen3vl-4b-r64-2x3090-20260922-235301/checkpoint-2960 \
DATASET=data/coco/train_sft_10pct_1to2to4_11829_inst-v5.json \
DATA_ROOT=data \
OUTPUT_DIR=outputs/train/grpo/qwen3vl-4b-sft-grpo \
bash scripts/train_grpo.sh
```

For FSDP full sharding across 2 or 4 GPUs, enable it explicitly.
FSDP shards model parameters, gradients, and optimizer states; each rank still
processes its own prompt. Keep the image pixel limits unchanged if you need the
original resolution:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
FSDP_MODE=full_shard \
MODEL_NAME_OR_PATH=weights/Qwen3-VL-4B-Instruct \
ADAPTER_PATH=outputs/train/sft/qwen3vl-4b-r64-2x3090-20260922-235301/checkpoint-2220 \
DATASET=data/coco/train_sft_10pct_1to2to4_11829_inst-v5.json \
DATA_ROOT=data \
OUTPUT_DIR=outputs/train/grpo/qwen3vl-4b-sft-grpo-fsdp4 \
NUM_GENERATIONS=4 \
MAX_PROMPT_LENGTH=4096 \
MAX_COMPLETION_LENGTH=512 \
MIN_PIXELS=4096 \
MAX_PIXELS=640000 \
bash scripts/train_grpo.sh
```

For 2 GPUs, set `CUDA_VISIBLE_DEVICES=0,1`, `NPROC_PER_NODE=2`, and use a
different output directory. FSDP can reduce parameter memory, but it does not
shard per-prompt image activations or the full-sequence logits; unusually large
multi-image prompts can still exceed one card's memory. Generation also gathers
FSDP units repeatedly, so expect it to run more slowly than ordinary DDP.

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
