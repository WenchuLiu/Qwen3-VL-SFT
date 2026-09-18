# Architecture

The project is organized around four runtime boundaries, following the
package layout of `../locateanything` without copying its model-specific
implementation.

```text
qwen3vl_sft/
├── model/
│   └── loader.py                 # checkpoint, processor, LoRA/full tuning
├── data/
│   ├── schema.py                 # public annotation validation
│   ├── messages.py               # Qwen record -> HF messages
│   ├── processing.py             # one record -> tensors and labels
│   └── rope.py                   # multimodal position IDs
├── train/
│   ├── arguments.py              # CLI and data defaults
│   ├── dataset.py                # annotation loading and splitting
│   ├── collator.py               # padding and vision tensor merging
│   ├── data.py                   # Trainer data-module assembly
│   ├── trainer.py                # public Trainer boundary
│   └── runner.py                 # end-to-end training orchestration
└── evaluation/
    ├── coco/                     # fixed protocol, episodes, generation, metrics
    └── fewshot/                  # cross-domain benchmark application
```

The dependency direction is intentionally one-way:

```text
arguments -> runner -> {model, train.data, train.trainer}
train.data -> {data.processing, data.messages, data.rope}
evaluation.coco -> {data.schema, no training runtime}
evaluation.fewshot -> evaluation.coco
```

The flat modules (`config.py`, `modeling.py`, `data/preprocess.py`, and the
original evaluation modules) are compatibility facades. New code should use
the package paths above. This keeps old experiment scripts working while
making each responsibility discoverable from the directory tree.

## Runtime flow

1. `train.arguments` parses one explicit experiment configuration.
2. `train.runner` validates the evaluation mode and creates the model,
   processor, datasets, collator, and Trainer.
3. `train.dataset` loads records; `data.processing` renders one multimodal
   conversation and creates assistant-only labels.
4. `train.trainer` runs loss evaluation or the fixed COCO generation protocol.
5. `model.loader` saves the model/adapter and processor side by side.

The COCO prompt is defined once in `evaluation/coco/protocol.py`. Dataset
builders, standalone evaluation, and in-training evaluation all consume that
same protocol.
