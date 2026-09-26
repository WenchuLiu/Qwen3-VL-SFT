#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-LLM}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
# Use any official CUDA-enabled PyTorch wheel that matches the host driver/GPU.
# cu118 is the conservative default; override this URL for another CUDA line.
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"
REQUIRE_CUDA="${REQUIRE_CUDA:-0}"

if ! command -v conda >/dev/null 2>&1; then
  echo "错误: 未找到 conda，请先加载 Miniconda/Anaconda。" >&2
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

python - <<'PY'
import importlib.util
import os
from importlib.metadata import version

import accelerate
import deepspeed
import peft
import torch
import transformers
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

pycocotools_version = version("pycocotools")

print(f"torch: {torch.__version__}; torch CUDA: {torch.version.cuda}")
print(f"transformers: {transformers.__version__}")
print(f"accelerate: {accelerate.__version__}")
print(f"deepspeed: {deepspeed.__version__}; peft: {peft.__version__}")
print(
    f"pycocotools: {pycocotools_version}; "
    f"runtime imports: {COCO.__name__}, {COCOeval.__name__}"
)
print(f"flash-attn installed: {importlib.util.find_spec('flash_attn') is not None}")

if not torch.cuda.is_available():
    message = "PyTorch CUDA 不可用；安装完成，但当前环境不能进行 GPU 训练/推理。"
    if os.environ.get("REQUIRE_CUDA", "0") == "1":
        raise RuntimeError(message)
    print(f"WARNING: {message}")
else:
    device_count = torch.cuda.device_count()
    print(f"CUDA devices: {device_count}")
    for index in range(device_count):
        name = torch.cuda.get_device_name(index)
        capability = torch.cuda.get_device_capability(index)
        print(
            f"GPU {index}: {name}; compute capability: "
            f"{capability[0]}.{capability[1]}"
        )
    try:
        bf16_supported = torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if bf16_supported else torch.float16
        x = torch.randn(1024, 1024, device="cuda", dtype=dtype)
        y = x @ x
        torch.cuda.synchronize()
        assert torch.isfinite(y).all().item(), "CUDA matrix multiplication check failed"
        print(f"CUDA {dtype} matrix multiplication: OK")
        print(f"CUDA BF16 supported: {bf16_supported}")
    except RuntimeError as error:
        message = f"CUDA 计算检查失败: {error}"
        if os.environ.get("REQUIRE_CUDA", "0") == "1":
            raise RuntimeError(message) from error
        print(f"WARNING: {message}")
PY

echo "LLM 环境安装并验证完成。使用: conda activate ${ENV_NAME}"
