# Python command-line tools

The Python implementations live under `qwen3vl_sft/`. Files in this directory
are thin command-line entry points for compatibility and convenient direct
execution:

- `train.py` -> `python -m qwen3vl_sft.train`
- `train_grpo.py` -> `python -m qwen3vl_sft.train.grpo`
- `evaluate_coco.py` -> fixed COCO generation evaluation
- `evaluate_fewshot.py` -> cross-domain few-shot application
- `build_coco.py` -> deterministic COCO records and episode manifests
- `compare.py` -> compare two compatible result files

Do not add prompt, model, or metric implementations here. Put reusable logic in
the package and keep this directory limited to CLI argument forwarding.
