# Qwen3-VL-SFT 环境安装（无需 FlashAttention）

本文档给出当前仓库的可复现安装方式。项目默认使用 PyTorch 的
`sdpa`（Scaled Dot Product Attention），不要求安装、编译或导入
`flash-attn`。

## 环境要求

- Linux
- Conda 或 Miniconda
- NVIDIA GPU；仓库提供的自动安装脚本按 NVIDIA A40、CUDA 11.8 测试
- Python 3.10
- 建议使用 BF16；A40 的计算能力为 `sm_86`

如果只做 CPU 代码检查，可以使用手动安装方式，但 Qwen3-VL 推理和训练
仍然需要 CUDA GPU。

## 推荐安装：Conda + CUDA 11.8

在已经分配 NVIDIA GPU 的计算节点上，从仓库根目录执行：

```bash
INSTALL_FLASH_ATTN=0 bash install_llm_env.sh
conda activate LLM
```

脚本会创建或复用 `LLM` 环境，并安装当前项目使用的固定版本：

- Python 3.10
- PyTorch 2.6.0、TorchVision 0.21.0、TorchAudio 2.6.0
- Transformers 4.57.1、Accelerate 1.11.0、PEFT 0.17.1
- `qwen-vl-utils`、DeepSpeed、SwanLab、ModelScope、COCO evaluation 依赖
- PyTorch CUDA 11.8 runtime

脚本默认已经将 `INSTALL_FLASH_ATTN` 设为 `0`；显式写出来是为了避免
误解。它不会下载或编译 FlashAttention，模型加载和评测统一使用
`sdpa`。脚本最后会检查 CUDA、BF16 矩阵乘法以及关键 Python 包。

该自动脚本会检查 `nvidia-smi` 和 A40 型号。如果当前是登录节点、没有
GPU，或者 GPU 不是 A40，请使用下面的手动安装方式，或切换到已分配的
GPU 节点。

## 手动安装：不使用 FlashAttention

如果不需要自动配置 A40 的 Conda CUDA 工具链，可以手动创建环境：

```bash
conda create -n qwen3-vl-sft python=3.10 -y
conda activate qwen3-vl-sft

python -m pip install --upgrade pip setuptools wheel packaging ninja psutil

python -m pip install \
  torch==2.6.0 \
  torchvision==0.21.0 \
  torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu118 \
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

python -m pip install -e . --no-deps
```

上面的命令没有 `flash-attn`，也没有使用 `--no-build-isolation` 去编译
FlashAttention。若某个 CUDA 扩展在特定机器上需要本地编译器，优先使用
上面的自动安装脚本；这不改变项目对 FlashAttention 的非依赖性。

## 验证安装

```bash
conda activate qwen3-vl-sft  # 或 conda activate LLM

python - <<'PY'
import torch
import transformers
import accelerate
import peft
import qwen_vl_utils

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)
print("peft:", peft.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

python -m qwen3vl_sft.train --help >/dev/null
python tools/evaluate_fewshot.py --help >/dev/null
```

如果需要确认环境中没有安装 FlashAttention，可以执行：

```bash
python - <<'PY'
import importlib.util
print("flash_attn installed:", importlib.util.find_spec("flash_attn") is not None)
PY
```

这里显示 `False` 是预期结果；即使机器上已有该包，只要使用 `sdpa`，本
项目也不会依赖它。

## 运行时使用 SDPA

训练、普通 COCO eval 和 VE eval 都默认使用 `sdpa`。可以显式写出配置：

```bash
ATTN_IMPLEMENTATION=sdpa bash scripts/train_lora.sh
ATTENTION=sdpa bash fewshot_eval/scripts/run_ve.sh
```

直接调用 Python 入口时使用：

```bash
torchrun --nproc_per_node=2 -m qwen3vl_sft.train \
  --model-name-or-path Qwen/Qwen3-VL-4B-Instruct \
  --dataset data/train.json \
  --output-dir runs/qwen3vl-lora \
  --attn-implementation sdpa
```

不要传 `--attn-implementation flash_attention_2` 或
`--attention flash_attention_2`。如果显存不足，优先降低
`--max-pixels`、`BATCH_SIZE` 或 `--per-device-train-batch-size`。

## 安装脚本的可选开关

FlashAttention 不是项目必需项，但为了兼容已有实验环境，安装脚本仍保留
显式开关。只有在明确需要时才使用：

```bash
INSTALL_FLASH_ATTN=1 bash install_llm_env.sh
```

不设置该变量，或设置为 `0`，都不会安装 FlashAttention。
