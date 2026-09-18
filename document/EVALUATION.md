# Evaluation

The evaluation package has two applications:

```text
qwen3vl_sft/evaluation/coco/
  protocol.py    prompt and episode schema
  data.py        COCO annotation -> train/eval records
  generation.py model.generate() and result payloads
  metrics.py    parsing, IoU/F1, and COCO mAP

qwen3vl_sft/evaluation/fewshot/
  config.py      dataset registry and cache validation
  cli.py         persistent single/multi-GPU benchmark runner
```

Build a fixed COCO manifest once and reuse it for base and adapter evaluation:

```bash
bash scripts/build_coco_eval.sh
MODEL_NAME_OR_PATH=Qwen/Qwen3-VL-4B-Instruct \
EPISODES=data/coco/val_episodes.json \
OUTPUT=runs/base.json \
bash scripts/evaluate_coco.sh
```

The five standard cross-domain datasets use 1,024 generated tokens;
VISUALDIOR uses 2,048. Their per-dataset launchers are under
`fewshot_eval/scripts/`, while the implementation and registry live under
`qwen3vl_sft/evaluation/fewshot/`.

Every result records the protocol version, episode hash, preprocessing budget,
generation budget, raw responses, parsed boxes, F1, and official COCO mAP.
This makes a before/after comparison auditable and prevents silently comparing
different prompts or manifests.
