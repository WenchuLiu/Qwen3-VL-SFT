# COCO inst-v5 0-shot SFT + GRPO on four RTX 3090 GPUs

The experiment starts from `weights/Qwen3-VL-4B-Instruct`, trains a new LoRA
adapter for three SFT epochs, then updates that adapter for 1,000 GRPO optimizer
steps. It saves and retains GRPO checkpoints at steps 250, 500, 750 and 1,000.
COCO evaluation runs after every SFT epoch and every 250 GRPO steps. After SFT
and at GRPO steps 500 and 1,000, a zero-shot evaluation runs on ArTaxOr,
Clipart1k, FISH, NEU-DET and UODD. VISUALDIOR is excluded.

The zero-shot manifest is derived from the previous inst-v5 11,829-record
1/2/4-shot manifest. Query images, category schedule, final ground-truth answers
and order are preserved; support images and demonstration turns are removed.
The existing seed-42 10% pool contains 11,829 image IDs. As in the previous
category-balanced query schedule, sampling with replacement yields 6,681 unique
query images in 11,829 records. Both SFT and GRPO use this same derived manifest.
SFT uses score-free ground-truth answers. GRPO removes the final answer from the
prompt, uses its boxes only for reward calculation, and asks for confidence
scores using the zero-shot inst-v5 evaluation prompt.

| Setting | SFT | GRPO |
| --- | --- | --- |
| GPUs | 4 DDP workers | 4 DDP workers |
| Precision / attention | BF16 / SDPA | BF16 / SDPA |
| LoRA | r64, alpha128, dropout0.05 | Continue SFT adapter; dropout disabled |
| Batch | 2 per GPU, accumulate2 (global16) | 1 prompt per GPU, 4 samples per prompt |
| Learning rate | 5e-5 | 1e-5 |
| Schedule | Cosine, warmup3% | Cosine, warmup3% |
| Duration | 3 epochs | 1,000 optimizer steps |
| Checkpoints | Every epoch; retain3 | Every250 steps; retain4 |
| Pixel budget | 4,096–640,000 | 4,096–640,000 |
| Sequence budget | 4,096 | Prompt4,096 / response512 |
| Reward / KL | — | BoxIoU + score + format; KL0.04, clip0.2 |

Both stages are evaluated on the same existing
`val_episodes_500_0shot_inst-v5.json` with deterministic decoding and a
1,024-token generation budget. Four independent GPU replicas shard these
500 category-conditioned tasks. Responses are restored to manifest order before
computing the repository's official COCO AP@[0.50:0.05:0.95] and F1 averaged over
the same ten IoU thresholds. This is the existing fixed-task evaluation protocol,
not a new full-val2017 all-category benchmark.

```bash
/root/miniconda3/envs/LLM/bin/python tools/prepare_coco_zero_shot.py

# Small end-to-end check; its metrics are not final experiment results.
/root/miniconda3/envs/LLM/bin/python tools/run_coco_zero_shot_sft_grpo.py \
  --run-dir "outputs/train/zero-shot-inst-v5-smoke-$(date +%Y%m%d-%H%M%S)" --smoke

# Full experiment, sequential SFT → eval → GRPO → eval.
/root/miniconda3/envs/LLM/bin/python tools/run_coco_zero_shot_sft_grpo.py \
  --run-dir outputs/train/qwen3vl-4b-0shot-inst-v5-sft3-grpo1000-20260930

# Initialize fresh LoRA on the original 4B model and run GRPO directly.
/root/miniconda3/envs/LLM/bin/python tools/run_coco_zero_shot_sft_grpo.py \
  --grpo-from-base \
  --run-dir outputs/train/qwen3vl-4b-base-0shot-inst-v5-lora-grpo1000-20261001
```

`--grpo-from-base` skips SFT and its evaluations. It creates a fresh LoRA
adapter on the original model, with the initial zero-update adapter as the
frozen KL reference. It keeps the same 10% zero-shot training data, 1,000-step
scheduler, checkpoints and COCO evaluations every 250 steps, and five-dataset
cross-domain zero-shot evaluations every 500 steps. Use a new run directory;
completed checkpoints in that directory are resumed only for that experiment.

The run directory contains `config.json` (including input hashes), `status.json`,
stage logs, SFT/GRPO adapters and checkpoints, `sft_epoch_metrics.json`, complete
`sft_coco_eval/step-*_0shot.json`, `sft_eval.json`,
`grpo_eval_{250,500,750}.json`, and `grpo_eval.json` predictions,
and compact `metrics.json` / `RESULTS.md` summaries. Cross-domain evaluations
live under `sft_cross_domain`, `grpo_cross_domain_500` and
`grpo_cross_domain_1000`, with individual predictions, manifests and summary
files. `cross_domain_metrics.json` collects mAP and average F1 per dataset.
The SFT summary is written immediately after SFT evaluation, before GRPO starts.
`sft_eval.json` copies the third epoch's evaluation; SFT runs exactly one COCO
evaluation per epoch and then proceeds to the five cross-domain evaluations.
`status.json` says `completed` / `all` only after all scheduled evaluations finish.

GRPO runs four recoverable segments with `--max-steps 1000` throughout and
`--stop-after-step 250/500/750/1000`. Each segment saves optimizer, scheduler,
RNG, policy and frozen reference state. Evaluation runs while training workers
are stopped, then the next segment resumes the checkpoint. Keeping the same
1,000-step scheduler horizon prevents LR resets between segments. The GRPO
checkpoint loader explicitly restores the root policy adapter along with the
`grpo_reference/` adapter, working around Transformers 4.57's subdirectory-only
adapter restoration when both layouts exist.

Re-running the same command skips completed stages and resumes an interrupted
training stage from its latest saved checkpoint. A lock prevents two pipeline
instances from writing the same run. Input/configuration changes require a new
run directory. A failed stage records its error and does not start later stages.

Validation completed before the full launch: four-GPU SFT/GRPO short runs,
COCO generation, five cross-domain zero-shot smoke evaluations, and recovery
from GRPO step 1 to step 2. The frozen reference tensors were identical across
that recovery, policy tensors changed, and the scheduler advanced without
resetting. The SFT online-evaluation export retained all eight smoke responses
and exactly matched the logged mAP and F1. The 20 relevant GRPO/zero-shot tests
and eight telemetry tests passed. The full repository test run also exposed an
existing two-GPU launcher expectation mismatch (the test expects batch2 /
accumulation4; that launcher uses batch1 / accumulation8, both global batch16).
