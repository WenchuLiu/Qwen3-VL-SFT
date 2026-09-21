# Qwen3-VL-SFT

Official implementation for Qwen3-VL continual SFT and cross-domain
few-shot detection evaluation. The repository provides ready-to-run scripts
for SFT, baseline evaluation, IE/VE evaluation, and DetPO evaluation.

## 🚀 Installation

在仓库根目录执行：

```bash
ENV_NAME=LLM bash install_llm_env.sh
conda activate LLM
```

安装脚本会创建 Python 3.10 环境，并安装项目依赖。项目默认使用
PyTorch `sdpa`，不要求安装 FlashAttention。需要 Linux、Conda 和可用的
NVIDIA CUDA 环境。

## ⚡ Quick Start

准备以下文件：

- 基础模型：`weights/Qwen3-VL-4B-Instruct`
- 六个跨域数据集：放在 `data/` 下（ArTaxOr、clipart1k、FISH、NEU-DET、UODD、VISUALDIOR）
- DetPO prompt：仓库已提供在 `docs/cross-domain-instructions/`

直接运行 DetPO 的 1/2/4-shot 评测：

```bash
DETPO=1 \
MODEL_PATH=weights/Qwen3-VL-4B-Instruct \
DATA_ROOT=data \
SHOTS="1 2 4" \
NUM_GPUS=4 \
bash scripts/evaluate_fewshot.sh
```

结果默认保存在：
`outputs/eval/fewshot/qwen3-vl-4b-detpo-fewshot/`。

## 🏋️ Training (Continual SFT)

训练数据使用 Qwen-VL conversation 格式，可以是 JSON 或 JSONL。使用
LoRA 进行 continual SFT：

```bash
MODEL_NAME_OR_PATH=weights/Qwen3-VL-4B-Instruct \
DATASET=data/train_sft.json \
OUTPUT_DIR=outputs/train/continual-sft \
NPROC_PER_NODE=4 \
REPORT_TO=none \
bash scripts/train_lora.sh
```

继续训练时，将 `DATASET` 换成新的数据文件，并将 `OUTPUT_DIR` 指向新的
实验目录。需要记录到 SwanLab 时，设置：

```bash
REPORT_TO=swanlab SWANLAB_API_KEY=<your-key> \
MODEL_NAME_OR_PATH=weights/Qwen3-VL-4B-Instruct \
DATASET=data/train_sft.json \
OUTPUT_DIR=outputs/train/continual-sft \
bash scripts/train_lora.sh
```

## 📈 Evaluation

所有评测脚本都从仓库根目录执行。`DATASETS` 和 `SHOTS` 使用空格分隔。

### DetPO eval

DetPO 会按照 dataset 和 shot 自动选择：
`{shot}-shot/all_refined_class_instructions_{dataset}.json`。

```bash
DETPO=1 \
MODEL_PATH=weights/Qwen3-VL-4B-Instruct \
DATA_ROOT=data \
SHOTS="1 2 4" \
NUM_GPUS=4 \
bash scripts/evaluate_fewshot.sh
```

只评测一个数据集：

```bash
DETPO=1 \
DATASETS=FISH \
SHOTS="1 2 4" \
MODEL_PATH=weights/Qwen3-VL-4B-Instruct \
DATA_ROOT=data \
NUM_GPUS=1 \
bash scripts/evaluate_fewshot.sh
```

自定义 prompt 目录：

```bash
DETPO=1 DETPO_PROMPTS=/path/to/cross-domain-instructions \
bash scripts/evaluate_fewshot.sh
```

### Baseline / IE / VE

Baseline：

```bash
MODEL_PATH=weights/Qwen3-VL-4B-Instruct \
DATA_ROOT=data \
SHOTS="0 1 2 4" \
NUM_GPUS=4 \
bash scripts/evaluate_fewshot.sh
```

Instruction Enhancement（IE）：

```bash
IE=1 \
CATEGORY_DESCRIPTIONS=docs/cross_domain_category_descriptions.json \
MODEL_PATH=weights/Qwen3-VL-4B-Instruct \
DATA_ROOT=data \
SHOTS="0 1 2 4" \
NUM_GPUS=4 \
bash scripts/evaluate_fewshot.sh
```

Visual Enhancement（VE）：

```bash
VE=1 \
MODEL_PATH=weights/Qwen3-VL-4B-Instruct \
DATA_ROOT=data \
SHOTS="1 2 4" \
NUM_GPUS=4 \
bash scripts/evaluate_fewshot.sh
```

DetPO 不支持 0-shot，但可以与 VE 组合；DetPO 和 IE 不能同时启用。
评测结果包含 `mAP@50:95`、`AP50`、`AP75`、预测框和原始模型输出。
使用 `SKIP_EXISTING=1` 可以跳过已完成的结果。

### COCO eval

```bash
COCO_ROOT=data/coco bash scripts/build_coco_train.sh
COCO_ROOT=data/coco bash scripts/build_coco_eval.sh

MODEL_NAME_OR_PATH=weights/Qwen3-VL-4B-Instruct \
EPISODES=data/coco/val_episodes.json \
OUTPUT=outputs/eval/coco/base/result.json \
bash scripts/evaluate_coco.sh
```

## 🙏 Acknowledgement

本项目基于以下开源项目：

- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [qwen-vl-utils](https://github.com/QwenLM/Qwen3-VL)

感谢相关作者和社区的开源工作。

## 📜 License

本项目采用 [Apache License 2.0](LICENSE) 发布。第三方模型、数据集和代码
请同时遵守其各自的许可证和使用条款。
