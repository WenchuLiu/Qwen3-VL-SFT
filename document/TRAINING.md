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

The GRPO launcher defaults to DDP across all four GPUs on the 3090server. DDP
keeps one complete model replica on each GPU and synchronizes gradients; it does
not split model parameters across GPUs. The 4B model and its per-rank activations
must therefore fit on one 24 GB card. For this Qwen3-VL run, keep image pixel
limits and prompt/completion lengths sized for that per-GPU memory:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
MODEL_NAME_OR_PATH=weights/Qwen3-VL-4B-Instruct \
ADAPTER_PATH=outputs/train/sft/qwen3vl-4b-r64-2x3090-20260922-235301/checkpoint-2220 \
DATASET=data/coco/train_sft_10pct_1to2to4_11829_inst-v5.json \
DATA_ROOT=data \
OUTPUT_DIR=outputs/train/grpo/qwen3vl-4b-sft-grpo-ddp4 \
ROLLOUT_BATCH_SIZE=4 \
ROLLOUT_N=4 \
MAX_PROMPT_LENGTH=4096 \
MAX_RESPONSE_LENGTH=512 \
MIN_PIXELS=4096 \
MAX_PIXELS=640000 \
bash scripts/train_grpo.sh
```

GRPO retains the full prompt for multimodal position handling but projects LM-head
logits only for response positions. The rollout settings follow Rex-Omni's naming:
ROLLOUT_BATCH_SIZE is the global number of prompts in one DDP rollout batch;
with four ranks, a value of 4 gives one prompt per rank. ROLLOUT_N is the number
of responses sampled per prompt (Rex-Omni worker.rollout.n), and
MAX_RESPONSE_LENGTH is the generation token budget (default 512). The old
NUM_GENERATIONS and MAX_COMPLETION_LENGTH environment names remain supported.
The GRPO launcher saves a Trainer checkpoint every 1000 optimizer steps by
default; set SAVE_STEPS to change the interval, SAVE_STRATEGY to choose another
policy, and SAVE_TOTAL_LIMIT to control checkpoint retention.
PER_DEVICE_TRAIN_BATCH_SIZE can still be set for legacy runs; then the global
rollout batch is derived as per-device batch size times DDP world size.

For 2 GPUs, set `CUDA_VISIBLE_DEVICES=0,1`, `NPROC_PER_NODE=2`, and use a
different output directory. Each DDP rank still needs enough memory for the
complete model, generated sequences, image activations, and logits; reducing the
GPU count does not reduce per-rank model memory. Use `NPROC_PER_NODE=1` only for
a single-GPU debug run.

`qwen3vl_sft.train.grpo_data` accepts the repository's SFT conversation
records, COCO episode records with `support`/`query`, and generic records with
a conversational `prompt`. The final answer is not fed to the model during a
rollout. Its boxes are attached as reward metadata, while the final prompt is
rewritten to request `bbox_2d`, `label`, and a confidence `score` in `[0, 1]`.

The default registry is `box_iou score format`. `box_iou` follows Rex-Omni's
BoxIoU reward: for each prediction and each ground-truth box, find the best
same-class IoU independently, average those values into precision and recall,
then return their harmonic mean. It does not apply one-to-one assignment or an
IoU cutoff; `--iou-threshold` only controls matching in the separate score
reward. An empty prediction and empty target set receive BoxIoU reward 1.
The score reward gives matched detections their emitted score and unmatched
detections `1 - score`; missing score fields get zero score reward. Group
advantages are normalized over
`--rollout-n` responses and optimized with a clipped PPO surrogate; adjust
its interval with `CLIP_EPSILON` (default `0.2`). For LoRA, the KL reference is
a frozen copy of the starting adapter, so SFT knowledge remains the anchor.

The token log-probabilities are computed from the unwarped model distribution,
so GRPO requires `temperature=1`, `top_p=1`, and `top_k=0`. The trainer sets
these values explicitly for generation and disables dropout for rollout and
policy scoring.

This first GRPO path intentionally has no generation-based validation loop;
run the existing fixed-episode COCO evaluator after each saved checkpoint.
