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
OUTPUT=outputs/eval/coco/base/result.json \
bash scripts/evaluate_coco.sh
```

For the standard 500-query-image benchmark, use
`scripts/evaluate_coco_benchmark.sh`; it builds the fixed manifest when it is
missing and then delegates to the same canonical evaluator. The old
`scripts/cross_domain_datasets/run_coco.sh` is a compatibility wrapper.

The five standard cross-domain datasets use 1,024 generated tokens;
VISUALDIOR uses 2,048. The canonical cross-domain launcher is
`scripts/evaluate_fewshot.sh`; the implementation and registry live under
`qwen3vl_sft/evaluation/fewshot/`. Dataset-specific launchers under
`scripts/cross_domain_datasets/` are compatibility paths.

Visual Enhancement is implemented in the shared prompt builder. The dedicated
launcher covers ArTaxOr, Clipart1k, FISH, NEU-DET, UODD, and VISUALDIOR at
1/2/4 shots. Run it from the repository root; its default paths are relative
to that root. With
`--ve`, each support image is copied in memory and its normalized GT boxes are
drawn in red before vision processing; the query image remains unchanged. The
standalone four-GPU launcher is:

```bash
VE=1 \
MODEL_PATH=weights/Qwen3-VL-4B-Instruct \
DATA_ROOT=data NUM_GPUS=4 \
bash scripts/evaluate_fewshot.sh
```

Results record both `visual_enhancement` and the short compatibility key `ve`,
so baseline and VE cached results are not considered interchangeable by
`--skip-existing`.

Instruction Enhancement is available with `--instruction-enhancement` (or
`--ie`). Pass `--category-descriptions descriptions.json`, where the JSON maps
each category name to a visual description. The description is inserted into
both support and query questions; this is inference-time only and does not
modify the model or training data. Episode records may alternatively carry a
`category_description` field. The result stores the selected prompt variant
and the description-file SHA-256. IE can be combined with VE, but unlike VE it
also supports zero-shot episodes.

The repository provides `docs/cross_domain_category_descriptions.json`, which
covers all 57 categories in ArTaxOr, Clipart1k, FISH, NEU-DET, UODD, and
VISUALDIOR with domain-specific descriptions.

Every result records the protocol version, episode hash, preprocessing budget,
generation budget, raw responses, parsed boxes, F1, and official COCO mAP.
This makes a before/after comparison auditable and prevents silently comparing
different prompts or manifests.
