#!/usr/bin/env bash
# Single-node, four-GPU BF16 LoRA SFT. Defaults target Qwen3-VL-4B, not 8B.
# Run this script from the repository root; default paths are relative to it.
# Example: DATASET=data/coco/train_sft_10pct_1to4_11829.json bash scripts/train_lora_r64_4x3090.sh
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-weights/Qwen3-VL-4B-Instruct}"
DATASET="${DATASET:-data/coco/train_sft_10pct_1to2to4_11829_inst-v5.json}"
DATA_ROOT="${DATA_ROOT:-data}"
EVAL_EPISODES="${EVAL_EPISODES:-data/coco/val_episodes_500_124_inst-v5.json}"
MIN_PIXELS="${MIN_PIXELS:-4096}"
MAX_PIXELS="${MAX_PIXELS:-640000}"
RUN_ID="${RUN_ID:-qwen3vl-4b-r64-4x3090-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/sft/${RUN_ID}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=".:${PYTHONPATH:-}"

ARGS=(
  --model-name-or-path "${MODEL_NAME_OR_PATH}"
  --dataset "${DATASET}"
  --data-root "${DATA_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --run-name "${RUN_NAME:-$(basename "${OUTPUT_DIR}")}"
  --report-to "${REPORT_TO:-none}"
  --lora-enable true
  --lora-r 64
  --lora-alpha 128
  --lora-dropout "${LORA_DROPOUT:-0.05}"
  --bf16 true
  --fp16 false
  --tf32 true
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}"
  --gradient-checkpointing true
  --ddp-find-unused-parameters false
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-2}"
  --learning-rate "${LEARNING_RATE:-5e-5}"
  --num-train-epochs "${NUM_TRAIN_EPOCHS:-3}"
  --max-steps "${MAX_STEPS:--1}"
  --warmup-ratio "${WARMUP_RATIO:-0.03}"
  --lr-scheduler-type cosine
  --weight-decay "${WEIGHT_DECAY:-0.0}"
  --max-grad-norm 1.0
  --model-max-length "${MODEL_MAX_LENGTH:-4096}"
  --min-pixels "${MIN_PIXELS}"
  --max-pixels "${MAX_PIXELS}"
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS:-2}"
  --eval-mode generation
  --eval-strategy epoch
  --coco-eval-episodes "${EVAL_EPISODES}"
  --coco-eval-batch-size "${COCO_EVAL_BATCH_SIZE:-1}"
  --coco-eval-min-pixels "${COCO_EVAL_MIN_PIXELS:-${MIN_PIXELS}}"
  --coco-eval-max-pixels "${COCO_EVAL_MAX_PIXELS:-${MAX_PIXELS}}"
  --coco-eval-max-new-tokens "${COCO_EVAL_MAX_NEW_TOKENS:-1024}"
  --per-device-eval-batch-size 1
  --save-strategy "${SAVE_STRATEGY:-steps}"
  --save-steps "${SAVE_STEPS:-200}"
  --save-total-limit "${SAVE_TOTAL_LIMIT:-3}"
  --logging-steps "${LOGGING_STEPS:-10}"
  --resume-training false
  --seed "${SEED:-42}"
)
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
# Explicit CLI values take precedence over the preset, including evaluation.
ARGS+=("$@")
COMMAND=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --standalone --nnodes=1 --nproc_per_node=4
  -m qwen3vl_sft.train "${ARGS[@]}"
)

# Does not import torch, load a model, create outputs, or start a training job.
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

"${PYTHON_BIN}" - <<'PY'
import torch

count = torch.cuda.device_count()
if count < 4:
    raise SystemExit(
        f"This preset needs four visible CUDA GPUs; found {count}. "
        "Check CUDA_VISIBLE_DEVICES or use DRY_RUN=1 to inspect the command."
    )
for index in range(4):
    prop = torch.cuda.get_device_properties(index)
    print(f"GPU {index}: {prop.name}, {prop.total_memory / 2**30:.1f} GiB", flush=True)
PY

exec "${COMMAND[@]}"
