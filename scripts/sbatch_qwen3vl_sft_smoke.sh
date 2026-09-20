#!/usr/bin/env bash

### Slurm resource request: one node, four GPUs, three CPU cores per GPU.
#SBATCH -p gpu
#SBATCH --job-name=qwen3vl-sft-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH -D .
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# Batch shells do not necessarily load Conda's shell function automatically.
source ~/.bashrc
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-LLM}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_HOME="$(getent passwd "${USER}" | cut -d: -f6)"
SWANLAB_ENV_FILE="${SWANLAB_ENV_FILE:-${USER_HOME}/.config/qwen3vl-sft/swanlab.env}"

# Keep the API key out of this script. The env file should contain only, for
# example: export SWANLAB_API_KEY='...'. It is optional only when REPORT_TO=none.
if [[ -f "${SWANLAB_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${SWANLAB_ENV_FILE}"
fi

export SWANLAB_PROJ_NAME="${SWANLAB_PROJ_NAME:-Qwen3-VL-SFT}"
export SWANLAB_WORKSPACE="${SWANLAB_WORKSPACE:-wenchuliu3}"
export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
unset SWANLAB_PROJECT

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${ROOT_DIR}/weights/Qwen3-VL-4B-Instruct}"
DATASET="${DATASET:-${ROOT_DIR}/data/LLM/coco/smoke_train_10images.json}"
EVAL_EPISODES="${EVAL_EPISODES:-${ROOT_DIR}/data/LLM/coco/smoke_eval_10.json}"
RUN_ID="${RUN_ID:-qwen3vl-sft-smoke-10images-10eval-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/sft/${RUN_ID}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
REPORT_TO="${REPORT_TO:-swanlab}"

# Defaults reproduce the validated smoke test. For a real run, submit with
# MAX_STEPS=-1 NUM_TRAIN_EPOCHS=12 and replace DATASET/EVAL_EPISODES.
MAX_STEPS="${MAX_STEPS:-1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
MIN_PIXELS="${MIN_PIXELS:-3136}"
MAX_PIXELS="${MAX_PIXELS:-50176}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"

if [[ ! -f "${MODEL_NAME_OR_PATH}/config.json" ]]; then
  echo "model config not found: ${MODEL_NAME_OR_PATH}" >&2
  exit 1
fi
if [[ ! -f "${DATASET}" ]]; then
  echo "training dataset not found: ${DATASET}" >&2
  exit 1
fi
if [[ ! -f "${EVAL_EPISODES}" ]]; then
  echo "evaluation episodes not found: ${EVAL_EPISODES}" >&2
  exit 1
fi
if [[ "${REPORT_TO}" == *swanlab* && -z "${SWANLAB_API_KEY:-}" ]]; then
  echo "SWANLAB_API_KEY is not set; source ${SWANLAB_ENV_FILE} or export it before sbatch" >&2
  exit 1
fi

# Do not inherit the unavailable cluster proxy into SwanLab or training.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy

cd "${ROOT_DIR}"
mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "host=$(hostname)"
echo "python=$(python -V 2>&1)"
echo "gpus=${NPROC_PER_NODE}"
echo "dataset=${DATASET}"
echo "eval_episodes=${EVAL_EPISODES}"
echo "output=${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NPROC_PER_NODE - 1)))}" \
  python -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE}" \
  -m qwen3vl_sft.train \
  --model-name-or-path "${MODEL_NAME_OR_PATH}" \
  --dataset "${DATASET}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-name "${RUN_NAME:-qwen3vl-sft-smoke-10images-10eval}" \
  --report-to "${REPORT_TO}" \
  --lora-enable true \
  --bf16 true \
  --fp16 false \
  --tf32 true \
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
  --num-train-epochs "${NUM_TRAIN_EPOCHS}" \
  --max-steps "${MAX_STEPS}" \
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --per-device-eval-batch-size "${PER_DEVICE_EVAL_BATCH_SIZE:-1}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --learning-rate "${LEARNING_RATE:-2e-4}" \
  --model-max-length "${MODEL_MAX_LENGTH}" \
  --min-pixels "${MIN_PIXELS}" \
  --max-pixels "${MAX_PIXELS}" \
  --save-strategy "${SAVE_STRATEGY:-epoch}" \
  --logging-steps "${LOGGING_STEPS:-1}" \
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS:-0}" \
  --eval-mode generation \
  --eval-strategy "${EVAL_STRATEGY:-steps}" \
  --eval-steps "${EVAL_STEPS:-1}" \
  --coco-eval-episodes "${EVAL_EPISODES}" \
  --coco-eval-batch-size "${COCO_EVAL_BATCH_SIZE:-1}" \
  --coco-eval-min-pixels "${MIN_PIXELS}" \
  --coco-eval-max-pixels "${MAX_PIXELS}" \
  --coco-eval-max-new-tokens "${MAX_NEW_TOKENS}" \
  --resume-training false
