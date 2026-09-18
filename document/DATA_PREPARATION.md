# Data Preparation

## Generic SFT records

Use a JSON list/object or JSONL file with the official Qwen conversation
schema. For every user turn, `<image>` and `<video>` placeholders are consumed
in order from the record's `image` and `video` fields. The validator rejects
unreferenced media, unsupported roles, and empty turns before tokenization.

```json
{
  "image": "images/0001.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nLocate the red car."},
    {"from": "gpt", "value": "[{\"bbox_2d\":[100,200,500,800]}]"}
  ]
}
```

## COCO in-context records

Build training records and fixed evaluation episodes independently:

```bash
bash scripts/build_coco_train.sh
bash scripts/build_coco_eval.sh
```

The builder records the protocol name, prompt version, seed, source annotation
paths, and shot counts. Keep the generated episode manifest immutable after
evaluation begins; changing support images or query sampling creates a new
experiment.

`qwen3vl_sft/evaluation/coco/data.py` owns COCO parsing, normalized coordinate
conversion, category balancing, and deterministic support sampling. Prompt and
answer serialization remains in `coco/protocol.py`, so data generation and
runtime evaluation cannot silently drift apart.
