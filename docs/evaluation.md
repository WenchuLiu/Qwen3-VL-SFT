# Evaluation Contract

## Why the old values disagree

Hugging Face `Trainer.evaluate()` normally executes a forward pass with the
ground-truth labels. The resulting `eval_loss` measures next-token prediction
under teacher forcing. It does not call `model.generate()`, does not parse
predicted boxes, and cannot be directly compared with an object-detection F1.

The old baseline shell script performs a different operation: it builds
support/query messages, calls greedy generation, parses the response, filters
the requested category, matches boxes by IoU, and reports F1. These are both
valid measurements, but they answer different questions.

## This repository's contract

`--eval-mode loss` keeps the standard Trainer behavior and reports
`eval_loss`. `--eval-mode generation` replaces loss evaluation with fixed COCO
episode generation. Both standalone evaluation and in-training evaluation call
the same functions:

```text
qwen3vl_sft/evaluation/coco/protocol.py   -> prompt and target serialization
qwen3vl_sft/evaluation/coco/generation.py -> processor + generate()
qwen3vl_sft/evaluation/coco/metrics.py    -> parser + IoU matching + F1
```

An episode has this shape:

```text
system instruction
support image + question -> support answer
...
query image + question -> model generation
```

The optional `--instruction-enhancement`/`--ie` variant appends the target
category's visual description to each support and query question. Descriptions
are supplied as a JSON category-to-text mapping with
`--category-descriptions`; this is inference-time prompt context and requires
no training or weight updates.

The model never receives the query answer. The SFT record does contain that
answer, but `loss_mode=last_assistant` makes the collator supervise only the
last assistant turn. Support answers are context, not training targets.

COCO ground truth does not provide calibrated prediction confidence. Therefore
SFT prompts and answers use only `bbox_2d` and `label`; they do not include a
synthetic confidence target. Evaluation support answers keep that score-free
format, while the final generation query explicitly asks the model to estimate a
score for each predicted box.

## Metric definition

For every shot count, predictions are filtered to the episode category. Greedy
one-to-one matching is performed at IoU thresholds 0.50, 0.55, ..., 0.95. At
each threshold:

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * precision * recall / (precision + recall)
```

`f1_mean_over_iou` is the arithmetic mean of the ten threshold F1 values.
`count_accuracy` and `mean_matched_iou` are retained as diagnostic metrics.

## DetPO-style COCO mAP

The generation prompt also asks for detections in descending confidence order
and a numeric `score` for every box. Following DetPO, a missing score falls back
to `0.5`. The parsed detections are restored to original-image pixel coordinates,
converted from `xyxy` to COCO `xywh`, and evaluated with the official
`pycocotools.COCOeval` implementation.

Each shot summary contains `coco_map.model`, using the model-reported scores,
and `coco_map.ranking`, replacing scores by a linear 1.0-to-0.1 rank score as
DetPO's optional ranking rescorer does. `map_50_95`, `map_50`, and `map_75`
correspond to the first three standard COCO statistics. The primary Trainer
metric is `eval_coco_map` (one-shot model-score AP50:95).

The metric is computed over the category-conditioned episodes in the fixed
manifest. It uses COCO's AP algorithm, but it is not directly comparable to a
full-dataset detector benchmark unless the manifest covers the intended image
and category combinations, including negative queries.

The default image budget is 3,136 to 640,000 pixels. If you override
`--min-pixels` or `--max-pixels`, pass the same values to the standalone
evaluator; the episode manifest does not silently encode preprocessing
settings.
