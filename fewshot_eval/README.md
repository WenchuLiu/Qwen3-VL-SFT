# Compatibility launchers

The few-shot implementation lives in
`qwen3vl_sft/evaluation/fewshot/`. The canonical operational entry point is:

```bash
bash scripts/evaluate_fewshot.sh
VE=1 bash scripts/evaluate_fewshot.sh
```

The scripts in this directory remain only to keep older experiment commands
working. They delegate to the canonical launcher and should not contain new
evaluation logic.
