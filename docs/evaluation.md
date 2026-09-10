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
evaluation/coco_protocol.py  -> prompt and target serialization
evaluation/generation.py     -> processor + generate()
evaluation/metrics.py        -> parser + IoU matching + F1
```

An episode has this shape:

```text
system instruction
support image + question -> support answer
...
query image + question -> model generation
```

The model never receives the query answer. The SFT record does contain that
answer, but `loss_mode=last_assistant` makes the collator supervise only the
last assistant turn. Support answers are context, not training targets.

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

The default image budget is 3,136 to 640,000 pixels. If you override
`--min-pixels` or `--max-pixels`, pass the same values to the standalone
evaluator; the episode manifest does not silently encode preprocessing
settings.
