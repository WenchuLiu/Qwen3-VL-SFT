#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-LLM}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CUDA_VERSION="${CUDA_VERSION:-11.8}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
MAX_JOBS="${MAX_JOBS:-3}"

if ! command -v conda >/dev/null 2>&1; then
  echo "错误: 未找到 conda，请先加载 Miniconda/Anaconda。" >&2
  exit 1
fi

# This script must run inside a Slurm allocation on an A40 compute node.
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "错误: 当前终端未暴露 GPU。请先 salloc 申请 GPU，再 ssh 到分配的 gpu 节点。" >&2
  exit 1
fi

GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
if ! grep -qi "A40" <<<"${GPU_NAMES}"; then
  echo "错误: 当前分配的 GPU 不是 NVIDIA A40:" >&2
  printf '%s\n' "${GPU_NAMES}" >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "复用已有 Conda 环境: ${ENV_NAME}"
else
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi
# Some cluster base environments ship compiler deactivate hooks that read
# unset CONDA_BACKUP_* variables. Limit the workaround to Conda activation.
set +u
conda activate "${ENV_NAME}"

python -m pip install --upgrade \
  pip setuptools wheel packaging ninja psutil

# PyTorch 2.6 satisfies this repository's declared torch>=2.6 requirement.
# CUDA 11.8 supports A40 (Ampere, sm_86) while working with more cluster drivers
# than CUDA 12.4. Use NVIDIA's frozen label: the current unlabelled channel can
# resolve the old 11.8 meta-package to incompatible CUDA 12.x components.
conda install -y --override-channels --strict-channel-priority \
  -c "nvidia/label/cuda-${CUDA_VERSION}.0" \
  -c conda-forge \
  "cuda-nvcc=${CUDA_VERSION}" \
  "cuda-cudart-dev=${CUDA_VERSION}"
conda install -y --override-channels -c conda-forge \
  gcc_linux-64=11 \
  gxx_linux-64=11 \
  numpy=1.26.4 \
  pillow=12.0.0
set -u
python -m pip install \
  torch==2.6.0 \
  torchvision==0.21.0 \
  torchaudio==2.6.0 \
  --index-url "${TORCH_INDEX_URL}" \
  --extra-index-url https://pypi.org/simple

python -m pip install \
  transformers==4.57.1 \
  tokenizers==0.22.0 \
  accelerate==1.11.0 \
  deepspeed==0.16.9 \
  peft==0.17.1 \
  qwen-vl-utils==0.0.14 \
  swanlab==0.10.0 \
  modelscope==1.39.1 \
  timm==1.0.22 \
  liger-kernel==0.6.4 \
  sentencepiece==0.2.0 \
  einops==0.8.1 \
  decord==0.6.0 \
  av==14.2.0 \
  numpy==1.26.4 \
  scipy==1.15.3 \
  scikit-learn==1.7.2 \
  pillow==12.0.0 \
  pycocotools==2.0.10 \
  safetensors==0.6.2 \
  wandb==0.22.3 \
  protobuf==6.33.1 \
  opencv-python-headless==4.11.0.86 \
  pytest==8.4.2 \
  ruff==0.14.5 \
  --extra-index-url https://pypi.org/simple

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python -m pip install -e "${ROOT_DIR}" --no-deps

export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS
export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-cc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++"

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-${ROOT_DIR}/../flash_attn-2.7.2.post1+cu11torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}"
  if [[ -f "${FLASH_ATTN_WHEEL}" ]]; then
    python -m pip install "${FLASH_ATTN_WHEEL}"
  else
    python -m pip install flash-attn==2.7.2.post1 \
      --no-build-isolation \
      --extra-index-url https://pypi.org/simple
  fi
fi

python - <<'PY'
import importlib.util

import accelerate
import deepspeed
import peft
import torch
import transformers

assert torch.cuda.is_available(), "PyTorch 无法访问已分配的 GPU"
assert torch.version.cuda == "11.8", f"预期 PyTorch CUDA 11.8，实际为 {torch.version.cuda}"

name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
assert "A40" in name, f"预期 NVIDIA A40，实际为 {name}"
assert capability == (8, 6), f"预期 A40 计算能力 8.6，实际为 {capability}"

x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
assert torch.isfinite(y).all().item(), "CUDA BF16 运算检查失败"

print(f"GPU: {name}; compute capability: {capability[0]}.{capability[1]}")
print(f"torch: {torch.__version__}; torch CUDA: {torch.version.cuda}")
print(f"transformers: {transformers.__version__}")
print(f"accelerate: {accelerate.__version__}")
print(f"deepspeed: {deepspeed.__version__}; peft: {peft.__version__}")
print(f"flash-attn installed: {importlib.util.find_spec('flash_attn') is not None}")
print("CUDA BF16 matrix multiplication: OK")
PY

echo "LLM 环境安装并验证完成。使用: conda activate ${ENV_NAME}"
