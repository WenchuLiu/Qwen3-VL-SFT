# Results and Artifacts

Training writes the final model/adapter, processor files, Trainer state, and
`run_args.json` under `--output-dir`. Generation validation additionally
writes `generation_eval/step-*.json`.

Standalone COCO evaluation writes one result JSON containing:

```text
protocol
prompt_template_version
prompt_variant_version
episodes_sha256
min_pixels / max_pixels / max_new_tokens
metrics_by_shot
predictions   # raw response plus parsed and target boxes
```

The few-shot application uses an MMDetection-style evaluation directory:

```text
outputs/eval/fewshot/qwen3-vl-4b-base-fewshot/
├── config.json
├── summary.json
└── <dataset>/<shot>shot/
    ├── episodes.json
    └── result.json
```

`result.json` is reusable only when its protocol, prompt variant, and
generation budget match the requested run. It also records `visual_enhancement`,
`instruction_enhancement`, and the category-description file hash when IE is
enabled. Set `SKIP_EXISTING=1` to reuse complete entries; changing a prompt,
episode manifest, or token budget should produce a new result.
