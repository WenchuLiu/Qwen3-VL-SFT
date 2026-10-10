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

## Flexible 10% COCO training episodes

Build the three episode types from the same selected 10% image-ID manifest
used by COCO training:

```bash
bash scripts/build_flexible_coco_episodes.sh
```

The canonical image-ID manifest is checked in at
`docs/coco_train_image_ids_10pct_seed42.json`, so another checkout uses the
exact same pool. If the selected manifest path is missing, the builder
recreates it from `instances_train2017.json` with fraction `0.1` and pool seed
`42`. Set `COCO_ROOT` when the COCO directory is outside `data/COCO`; set
`ANNOTATIONS`, `IMAGE_ROOT`, or `IMAGE_IDS` to override individual paths.

The default output is `data/coco/train_episodes_10pct_flexible_11820.json`.
Its 11,820 records use the requested 50/30/20 split: 5,910 standard, 3,546
support-defined (1,773 paired queries), and 2,364 distractor episodes. Standard
episodes cycle through 0/1/2/4 shots; the other two types cycle through 1/2/4
shots. The count must be divisible by 20 so the ratios and paired
support-defined records remain exact.

The explicit upper-category map is in
`docs/coco_upper_category_map.json`. Its `vehicle` group currently contains
`car`, `bus`, and `truck`, explicitly instantiating the requested mapping;
manually review any additions before using them in a run. A
support-defined pair shares one query image and `pair_id`, while choosing a
different target subclass and matching support for each record. Distractor
records pair with standard records on the same query image, category, and shot
count.

Every record retains `query_image_id`, `support_image_ids`, `target_subclass`,
and `pair_id` for audits. The prompt builder uses each support frame's own
`category` for its question and answer, while the final query label follows the
episode's top-level `category`. Query boxes stay in the record for rewards and
are not included in the prompt. The prompt omits `episode_type` and applies the
same flexible-example instruction to all three types. GRPO can load this file
directly; its existing reward functions continue to score only the query
boxes and labels.

To use a different episode budget or map, set `NUM_EPISODES`, `OUTPUT`, or
`UPPER_CATEGORY_MAP` when running the script. The builder requires an image-ID
manifest with `fraction: 0.1`; it only samples four-shot episodes when four
distinct support images are available in that pool.

The episode JSON is a GRPO input and can be loaded directly by the trainer.
See [Score-aware GRPO](TRAINING.md#score-aware-grpo) for the combined build and
training command.
