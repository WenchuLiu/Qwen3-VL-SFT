# Qwen3-VL-SFT 环境安装（无需 FlashAttention）

本文档给出当前仓库的可复现安装方式。项目默认使用 PyTorch 的
`sdpa`（Scaled Dot Product Attention），不要求安装、编译或导入
`flash-attn`。

## 环境要求

- Linux
- Conda 或 Miniconda
- NVIDIA GPU；自动安装脚本默认使用 CUDA 11.8 的 PyTorch wheel，也支持通过
  `TORCH_INDEX_URL` 切换到其他官方 CUDA wheel
- Python 3.10
- 建议使用 BF16；是否支持 BF16 取决于目标 GPU

如果只做 CPU 代码检查，可以使用手动安装方式；Qwen3-VL 推理和训练仍然
建议使用 CUDA GPU。自动安装器不检查 GPU 型号，也不要求 A40 或固定的
`sm_86` 架构。

## 推荐安装：Conda + CUDA 11.8

在仓库根目录执行：

```bash
bash install_llm_env.sh
conda activate LLM
```

脚本会创建或复用 `LLM` 环境，并安装当前项目使用的固定版本：

- Python 3.10
- PyTorch 2.6.0、TorchVision 0.21.0、TorchAudio 2.6.0
- Transformers 4.57.1、Accelerate 1.11.0、PEFT 0.17.1
- `qwen-vl-utils`、DeepSpeed、SwanLab、ModelScope、COCO evaluation 依赖
- 默认 PyTorch CUDA 11.8 runtime；可通过 `TORCH_INDEX_URL` 选择其他官方
  CUDA wheel

它不会下载或编译 FlashAttention，模型加载和评测统一使用 `sdpa`。脚本
会枚举所有可见 GPU，并在 CUDA 可用时进行矩阵乘法检查；不会检查 GPU
必须是 A40，也不会固定 `sm_86`。

如果当前没有 GPU，安装仍可完成，但只能做 CPU 级别的导入和代码检查。
设置 `REQUIRE_CUDA=1` 可以让安装器在 CUDA 不可用时直接失败。

## 手动安装：不使用 FlashAttention

如果不需要自动配置 CUDA 相关环境，可以手动创建环境：

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
FlashAttention。若某个 CUDA 扩展在特定机器上需要本地编译器，请根据目标
GPU 和 CUDA 版本补充对应工具链；这不改变项目对 FlashAttention 的非依赖性。

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
如果目标 GPU 不支持 BF16，可用 `BF16=false FP16=true bash
scripts/train_lora.sh`，或在 Python 入口传 `--bf16 false --fp16 true`。

## 可选安装 FlashAttention

FlashAttention 不是本项目的必需依赖。只有在明确需要更快的 attention
kernel 时，才按照官方仓库的硬件、CUDA 和 PyTorch 兼容性说明单独安装：

```bash
python -m pip install flash-attn --no-build-isolation
```

官方来源：[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)。
安装前请以官方 README 的版本和平台要求为准；本项目默认仍使用 `sdpa`，
即使不安装 FlashAttention 也可以完成训练、普通 eval 和 VE eval。
