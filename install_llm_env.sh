#!/usr/bin/env bash
set -e

# 创建并进入环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n llm python=3.10 -y
conda activate llm

# 基础安装工具
pip install --upgrade pip setuptools wheel

# PyTorch 2.4.1 + CUDA 12.1
pip install \
  torch==2.4.1 \
  torchvision==0.19.1 \
  torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121

# LocateAnything / Qwen3-VL 当前环境的主要依赖
pip install \
  transformers==4.57.1 \
  tokenizers==0.22.0 \
  accelerate==1.5.2 \
  deepspeed==0.15.4 \
  peft==0.12.0 \
  qwen-vl-utils==0.0.14 \
  modelscope==1.39.1 \
  bitsandbytes==0.49.2 \
  datasets==5.0.0 \
  timm==1.0.27 \
  liger-kernel==0.3.1 \
  sentencepiece==0.2.0 \
  einops==0.8.2 \
  decord==0.6.0 \
  av==17.1.0 \
  numpy==1.26.4 \
  scipy==1.15.3 \
  scikit-learn==1.7.2 \
  pillow==12.2.0 \
  safetensors==0.8.0 \
  ninja==1.13.0 \
  wandb==0.28.0 \
  protobuf==7.35.1 \
  opencv-python-headless

# FlashAttention 需要系统已安装 CUDA Toolkit（可执行 nvcc）
pip install flash-attn==2.6.3 --no-build-isolation

# 输出关键版本进行检查
python -c "import torch, transformers, accelerate, deepspeed, peft; print('torch:', torch.__version__); print('torch cuda:', torch.version.cuda); print('transformers:', transformers.__version__); print('accelerate:', accelerate.__version__); print('deepspeed:', deepspeed.__version__); print('peft:', peft.__version__)"

echo "llm 环境安装完成"
