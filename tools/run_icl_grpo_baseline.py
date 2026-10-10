#!/usr/bin/env python3
"""Run the base-model ICL→GRPO baseline with resumable checkpoint evaluations.

Training is launched as short-lived DDP processes.  At each evaluation point
the process exits after saving a normal Trainer checkpoint, so generation
evaluation runs with all training workers and their CUDA allocations released.
Rerunning this command with the same run directory resumes the saved optimizer,
scheduler, RNG and dataloader state.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "weights" / "Qwen3-VL-4B-Instruct"
TRAIN_MANIFEST = ROOT / "data" / "coco" / "train_sft_10pct_1to2to4_11829_inst-v5.json"
COCO_MANIFEST_ROOT = ROOT / "data" / "coco"
CROSS_MANIFEST_ROOT = ROOT / "outputs" / "eval" / "fewshot" / "4b-sft-epoch4-6CD_dataset"
DATASETS = ("ArTaxOr", "Clipart1k", "FISH", "NEU-DET", "UODD")
SHOTS = (0, 1, 2, 4)
COCO_MANIFESTS = {
    shot: f"val_episodes_500_{shot}shot_inst-v5.json" for shot in SHOTS
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def copy_manifest(source: Path, destination: Path, *, max_records: int | None = None) -> dict:
    payload = read_json(source)
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"expected a manifest object with records: {source}")
    if max_records is not None:
        payload["records"] = payload["records"][:max_records]
        if not payload["records"]:
            raise ValueError(f"cannot make a non-empty smoke manifest from {source}")
        payload["smoke_subset_of_records"] = max_records
    write_json(destination, payload)
    return payload


def _validate_training_manifest(path: Path) -> tuple[dict, list[dict]]:
    payload = read_json(path)
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError(f"training manifest contains no records: {path}")
    if payload.get("prompt_template_version") != "inst-v5":
        raise ValueError("the GRPO baseline requires the inst-v5 training manifest")
    if len(records) != 11829:
        raise ValueError(f"expected 11,829 training records, found {len(records)}")
    shot_counts = {shot: 0 for shot in (1, 2, 4)}
    for index, record in enumerate(records):
        shot = record.get("num_shots")
        if shot not in shot_counts:
            raise ValueError(f"training record {index} has unexpected shot count {shot!r}")
        shot_counts[shot] += 1
    if any(count == 0 for count in shot_counts.values()):
        raise ValueError(f"the training manifest must retain the 1/2/4-shot mixture: {shot_counts}")
    return payload, records


def _validate_fixed_eval_manifests(coco_root: Path, cross_root: Path) -> dict:
    source_config = cross_root / "config.json"
    if not source_config.is_file():
        raise FileNotFoundError(f"fixed cross-domain evaluation config not found: {source_config}")
    config = read_json(source_config)
    if config.get("prompt_method") != "baseline" or config.get("detpo") or config.get("visual_enhancement") or config.get("instruction_enhancement"):
        raise ValueError("cross-domain manifests must be the baseline inst-v5 prompt set")
    if not config.get("all_query_pairs", False):
        raise ValueError("cross-domain fixed manifests must contain all query pairs")
    config_datasets = set(config.get("datasets", []))
    missing = sorted(set(DATASETS) - config_datasets)
    if missing:
        raise ValueError(f"cross-domain manifest source is missing datasets: {missing}")
    config_shots = set(config.get("shots", []))
    if not set(SHOTS).issubset(config_shots):
        raise ValueError(f"cross-domain manifest source lacks required shots: {sorted(config_shots)}")

    entries = {}
    for shot, filename in COCO_MANIFESTS.items():
        source = coco_root / filename
        if not source.is_file():
            raise FileNotFoundError(f"fixed COCO manifest not found: {source}")
        payload = read_json(source)
        records = payload.get("records", [])
        if payload.get("prompt_template_version") != "inst-v5" or not records:
            raise ValueError(f"invalid fixed COCO manifest: {source}")
        if any(record.get("num_shots") != shot for record in records):
            raise ValueError(f"COCO manifest has records from another shot count: {source}")
        entries[f"COCO/{shot}shot"] = {
            "source": str(source.resolve()),
            "sha256": sha256_file(source),
            "episodes": len(records),
            "shot": shot,
        }

    for dataset in DATASETS:
        for shot in SHOTS:
            source = cross_root / dataset / f"{shot}shot" / "episodes.json"
            if not source.is_file():
                raise FileNotFoundError(f"fixed cross-domain manifest not found: {source}")
            payload = read_json(source)
            records = payload.get("records", [])
            if payload.get("prompt_template_version") != "inst-v5" or not records:
                raise ValueError(f"invalid cross-domain manifest: {source}")
            support_histogram: dict[str, int] = {}
            for record in records:
                support_count = len(record.get("support", []))
                if record.get("num_shots") != shot or support_count > shot:
                    raise ValueError(f"fixed manifest has an invalid {shot}-shot record: {source}")
                support_histogram[str(support_count)] = support_histogram.get(str(support_count), 0) + 1
            entries[f"{dataset}/{shot}shot"] = {
                "source": str(source.resolve()),
                "sha256": sha256_file(source),
                "episodes": len(records),
                "shot": shot,
                "support_count_histogram": support_histogram,
            }
    return {"source_config": str(source_config.resolve()), "entries": entries}


def _smoke_training_payload(payload: dict) -> dict:
    """Select two examples per original ICL shot count without resampling."""
    selected = []
    for shot in (1, 2, 4):
        matches = [record for record in payload["records"] if record.get("num_shots") == shot]
        if len(matches) < 2:
            raise ValueError(f"smoke test needs two {shot}-shot records")
        selected.extend(matches[:2])
    return {**payload, "records": selected, "smoke_subset_of_records": len(selected)}


def _default_run_dir(smoke: bool) -> Path:
    tag = "icl-grpo-baseline-smoke" if smoke else "icl-grpo-baseline"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return ROOT / "outputs" / "train" / f"{tag}-{stamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--python", dest="python", default=sys.executable)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--dataset", type=Path, default=TRAIN_MANIFEST)
    parser.add_argument("--coco-manifest-dir", type=Path, default=COCO_MANIFEST_ROOT)
    parser.add_argument("--cross-domain-manifest-root", type=Path, default=CROSS_MANIFEST_ROOT)
    parser.add_argument("--smoke", action="store_true", help="short resumable DDP and generation-evaluation smoke run")
    parser.add_argument("--smoke-eval-interval", type=int, default=1)
    parser.add_argument("--smoke-manifest-records", type=int, default=1)
    return parser.parse_args()


def _effective_spec(args: argparse.Namespace, run_dir: Path, num_gpus: int, source_registry: dict, train_source: Path, model_path: Path) -> dict:
    if args.smoke:
        save_steps = 1
        eval_steps = args.smoke_eval_interval
        epochs = 1.0
        prompt_batch = num_gpus
        completion_length = 128
        train_record_count = 6
    else:
        save_steps = 250
        eval_steps = 500
        epochs = 1.0
        prompt_batch = 8
        completion_length = 512
        train_record_count = 11829
    return {
        "run_name": run_dir.name,
        "mode": "fresh_base_model_lora_grpo",
        "model": str(model_path),
        "initial_adapter": None,
        "sft": None,
        "training_manifest": {
            "source": str(train_source),
            "sha256": sha256_file(train_source),
            "prompt_template_version": "inst-v5",
            "source_records": 11829,
            "used_records": train_record_count,
            "shot_mix": "original 1/2/4-shot mix",
        },
        "training": {
            "num_train_epochs": epochs,
            "max_steps": -1,
            "total_optimizer_steps": None,
            "world_size": num_gpus,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "global_prompts_per_optimizer_step": prompt_batch,
            "rollouts_per_prompt": 4,
            "global_rollouts_per_optimizer_step": prompt_batch * 4,
            "learning_rate": 1e-5,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            "kl_coefficient": 0.04,
            "clip_epsilon": 0.2,
            "rewards": {"box_iou": 1.0, "score": 1.0, "format": 1.0},
            "lora": {"enabled": True, "rank": 64, "alpha": 128, "dropout": 0.05},
            "bf16": True,
            "gradient_checkpointing": True,
            "attention": "sdpa",
            "min_pixels": 4096,
            "max_pixels": 640000,
            "max_prompt_length": 8192,
            "max_response_length": completion_length,
            "save_every_optimizer_steps": save_steps,
            "save_total_limit": None,
            "eval_every_optimizer_steps": eval_steps,
            "seed": 42,
            "smoke_test": args.smoke,
        },
        "evaluation": {
            "datasets": ["COCO", *DATASETS],
            "shots": list(SHOTS),
            "cross_domain_manifest_source": str(Path(args.cross_domain_manifest_root).resolve()),
            "cross_domain_prompt_method": "baseline inst-v5",
            "fixed_manifest_registry": source_registry["entries"],
            "generation": {
                "deterministic": True,
                "max_new_tokens": 1024,
                "min_pixels": 4096,
                "max_pixels": 640000,
            },
            "smoke_manifest_records": args.smoke_manifest_records if args.smoke else None,
        },
    }


def _prepare_snapshots(run_dir: Path, train_source: Path, coco_root: Path, cross_root: Path, smoke: bool, smoke_records: int) -> tuple[Path, Path, dict]:
    manifest_root = run_dir / "manifests"
    train_snapshot = manifest_root / "train" / train_source.name
    coco_snapshot_root = manifest_root / "coco"
    cross_snapshot_root = manifest_root / "cross_domain"
    if not train_snapshot.is_file():
        source_payload = read_json(train_source)
        if smoke:
            source_payload = _smoke_training_payload(source_payload)
        write_json(train_snapshot, source_payload)
    elif sha256_file(train_source) != _read_spec_input_hash(run_dir, "training_manifest"):
        raise ValueError("training source manifest changed since this run was created")

    for shot, filename in COCO_MANIFESTS.items():
        source = coco_root / filename
        target = coco_snapshot_root / filename
        if not target.is_file():
            copy_manifest(source, target, max_records=smoke_records if smoke else None)
    for dataset in DATASETS:
        for shot in SHOTS:
            source = cross_root / dataset / f"{shot}shot" / "episodes.json"
            target = cross_snapshot_root / dataset / f"{shot}shot" / "episodes.json"
            if not target.is_file():
                copy_manifest(source, target, max_records=smoke_records if smoke else None)

    snapshot_registry = {"train": {}, "coco": {}, "cross_domain": {}}
    for path in [train_snapshot, *sorted(coco_snapshot_root.glob("*.json")), *sorted(cross_snapshot_root.glob("**/episodes.json"))]:
        relative = path.relative_to(manifest_root).as_posix()
        info = {"path": str(path.resolve()), "sha256": sha256_file(path)}
        if relative.startswith("train/"):
            snapshot_registry["train"] = info
        elif relative.startswith("coco/"):
            snapshot_registry["coco"][path.stem] = info
        else:
            shot = path.parent.name
            dataset = path.parent.parent.name
            snapshot_registry["cross_domain"][f"{dataset}/{shot}"] = info
    write_json(manifest_root / "manifest_registry.json", snapshot_registry)
    return train_snapshot, manifest_root, snapshot_registry


def _read_spec_input_hash(run_dir: Path, key: str) -> str | None:
    path = run_dir / "run_spec.json"
    if not path.is_file():
        return None
    spec = read_json(path)
    return spec.get(key, {}).get("sha256")


def _latest_checkpoint(grpo_dir: Path) -> Path | None:
    checkpoints = []
    if grpo_dir.is_dir():
        for path in grpo_dir.glob("checkpoint-*"):
            if not (path / "trainer_state.json").is_file():
                continue
            try:
                step = int(path.name.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                continue
            checkpoints.append((step, path))
    return max(checkpoints, default=(0, None))[1]


def _checkpoint_step(checkpoint: Path | None) -> int:
    return int(checkpoint.name.rsplit("-", 1)[1]) if checkpoint is not None else 0


def _training_state(grpo_dir: Path) -> dict | None:
    root_state = grpo_dir / "trainer_state.json"
    checkpoint = _latest_checkpoint(grpo_dir)
    checkpoint_state = checkpoint / "trainer_state.json" if checkpoint is not None else None
    candidates = [path for path in (root_state, checkpoint_state) if path is not None and path.is_file()]
    if not candidates:
        return None
    states = [read_json(path) for path in candidates]
    # A root TrainerState file can be newer than the most recent complete
    # checkpoint after an external interruption.  Prefer the checkpoint so a
    # resumed run always has matching optimizer/scheduler/RNG state.
    return max(states, key=lambda item: int(item.get("global_step", 0)))


def _epoch_complete(state: dict | None) -> bool:
    if not state:
        return False
    steps = int(state.get("global_step", 0))
    horizon = int(state.get("max_steps", 0))
    if horizon > 0 and steps >= horizon:
        return True
    epoch = state.get("epoch")
    return isinstance(epoch, (int, float)) and float(epoch) >= 1.0 - 1e-8


def _set_total_steps(run_dir: Path, state: dict) -> None:
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        return
    config = read_json(config_path)
    horizon = int(state.get("max_steps", 0))
    config["training"]["total_optimizer_steps"] = horizon if horizon > 0 else None
    config["training"]["last_global_step"] = int(state.get("global_step", 0))
    config["training"]["epoch_at_last_save"] = state.get("epoch")
    config["training"]["total_optimizer_steps_source"] = "TrainerState.max_steps for the 1-epoch dataloader"
    write_json(config_path, config)


def _child_env(root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "4")
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        env.pop(key, None)
    return env


def _write_status(run_dir: Path, *, state: str, stage: str, **details) -> None:
    write_json(
        run_dir / "status.json",
        {"state": state, "stage": stage, "updated_at": now(), **details},
    )


def _run_child(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path, run_dir: Path, stage: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _write_status(run_dir, state="running", stage=stage, command=command, log=str(log_path))
    print(f"{stage}: starting; log={log_path}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        _write_status(run_dir, state="running", stage=stage, child_pid=child.pid, command=command, log=str(log_path))
        code = child.wait()
    if code:
        _write_status(run_dir, state="failed", stage=stage, exit_code=code, log=str(log_path))
        raise RuntimeError(f"{stage} failed with exit code {code}; inspect {log_path}")
    _write_status(run_dir, state="completed", stage=stage, exit_code=0, log=str(log_path))


def _training_command(args: argparse.Namespace, *, run_dir: Path, train_snapshot: Path, target_step: int, resume: Path | None, num_gpus: int) -> list[str]:
    grpo_dir = run_dir / "grpo"
    completion_length = 128 if args.smoke else 512
    command = [
        args.python,
        "-m", "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={num_gpus}",
        "-m", "qwen3vl_sft.train.grpo",
        "--model-name-or-path", str(MODEL.resolve()),
        "--dataset", str(train_snapshot.resolve()),
        "--data-root", str((ROOT / "data").resolve()),
        "--output-dir", str(grpo_dir.resolve()),
        "--run-name", run_dir.name,
        "--report-to", "none",
        "--eval-mode", "none",
        "--eval-strategy", "no",
        "--bf16", "true",
        "--fp16", "false",
        "--tf32", "true",
        "--lora-enable", "true",
        "--lora-r", "64",
        "--lora-alpha", "128",
        "--lora-dropout", "0.05",
        "--attn-implementation", "sdpa",
        "--gradient-checkpointing", "true",
        "--ddp-find-unused-parameters", "false",
        "--min-pixels", "4096",
        "--max-pixels", "640000",
        "--model-max-length", "8192",
        "--max-prompt-length", "8192",
        "--max-response-length", str(completion_length),
        "--rollout-batch-size", str(num_gpus),
        "--rollout-n", "4",
        "--gradient-accumulation-steps", "1",
        "--learning-rate", "1e-5",
        "--warmup-ratio", "0.03",
        "--lr-scheduler-type", "cosine",
        "--num-train-epochs", "1",
        "--max-steps", "-1",
        "--save-strategy", "steps",
        "--save-steps", "1" if args.smoke else "250",
        "--logging-steps", "1" if args.smoke else "10",
        "--reward-functions", "box_iou", "score", "format",
        "--iou-reward-weight", "1",
        "--score-reward-weight", "1",
        "--format-reward-weight", "1",
        "--kl-coef", "0.04",
        "--clip-epsilon", "0.2",
        "--seed", "42",
        "--resume-training", "false",
        "--stop-after-step", str(target_step),
    ]
    # GRPO parser defaults save_total_limit=None. Omitting the option preserves
    # every numbered checkpoint in the run directory.
    if resume is not None:
        command.extend(["--resume-from-checkpoint", str(resume.resolve())])
    return command


def _evaluation_command(args: argparse.Namespace, *, run_dir: Path, snapshot_root: Path, checkpoint: Path, step: int, num_gpus: int) -> list[str]:
    return [
        args.python,
        str(ROOT / "tools" / "evaluate_icl_grpo_checkpoint.py"),
        "--model-path", str(MODEL.resolve()),
        "--adapter-path", str(checkpoint.resolve()),
        "--data-root", str((ROOT / "data").resolve()),
        "--coco-manifest-dir", str((snapshot_root / "coco").resolve()),
        "--cross-domain-manifest-root", str((snapshot_root / "cross_domain").resolve()),
        "--output-dir", str((run_dir / "eval" / f"step-{step:06d}").resolve()),
        "--num-gpus", str(num_gpus),
        "--batch-size", "1",
        "--min-pixels", "4096",
        "--max-pixels", "640000",
        "--max-new-tokens", "1024",
    ]


def _evaluation_complete(path: Path) -> bool:
    summary_path = path / "summary.json"
    if not summary_path.is_file():
        return False
    try:
        summary = read_json(summary_path)
        rows = summary.get("results", [])
        return len(rows) == (1 + len(DATASETS)) * len(SHOTS)
    except (OSError, json.JSONDecodeError, TypeError):
        return False


def _write_overall_summary(run_dir: Path) -> None:
    eval_root = run_dir / "eval"
    rows = []
    for summary_path in sorted(eval_root.glob("step-*/summary.json")):
        try:
            summary = read_json(summary_path)
        except (OSError, json.JSONDecodeError):
            continue
        step_dir = summary_path.parent
        step = int(step_dir.name.rsplit("-", 1)[1])
        for row in summary.get("results", []):
            rows.append({"checkpoint_step": step, **row})
    write_json(eval_root / "summary.json", {"results": rows})
    if rows:
        with (eval_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _run_evaluation(args: argparse.Namespace, *, run_dir: Path, snapshot_root: Path, checkpoint: Path, step: int, num_gpus: int, env: dict[str, str]) -> None:
    output_dir = run_dir / "eval" / f"step-{step:06d}"
    if _evaluation_complete(output_dir):
        print(f"evaluation step {step}: already complete", flush=True)
        return
    command = _evaluation_command(
        args,
        run_dir=run_dir,
        snapshot_root=snapshot_root,
        checkpoint=checkpoint,
        step=step,
        num_gpus=num_gpus,
    )
    _run_child(
        command,
        cwd=ROOT,
        env=env,
        log_path=run_dir / "logs" / f"eval-step-{step:06d}.log",
        run_dir=run_dir,
        stage=f"evaluate_step_{step}",
    )
    if not _evaluation_complete(output_dir):
        raise RuntimeError(f"evaluation at step {step} did not produce all result files")
    _write_overall_summary(run_dir)


def execute(args: argparse.Namespace, run_dir: Path) -> None:
    train_source = args.dataset.expanduser().resolve()
    coco_root = args.coco_manifest_dir.expanduser().resolve()
    cross_root = args.cross_domain_manifest_root.expanduser().resolve()
    model_path = MODEL.resolve()
    if not model_path.is_dir():
        raise NotADirectoryError(f"base model is missing: {model_path}")
    if not train_source.is_file():
        raise FileNotFoundError(f"training manifest is missing: {train_source}")
    if args.num_gpus < 2:
        raise ValueError("DDP baseline requires at least two GPUs")
    if not args.smoke and args.num_gpus != 8:
        raise ValueError("the formal baseline is defined for exactly eight DDP GPU replicas")
    if args.smoke and args.smoke_eval_interval < 1:
        raise ValueError("smoke-eval-interval must be positive")
    if args.smoke and args.smoke_manifest_records < 1:
        raise ValueError("smoke-manifest-records must be positive")

    # Import from the selected interpreter and fail before creating model jobs
    # if the requested DDP topology is not visible to this process.
    import torch

    visible_gpus = torch.cuda.device_count()
    if not torch.cuda.is_available() or visible_gpus < args.num_gpus:
        raise RuntimeError(
            f"requested {args.num_gpus} GPUs, but this process sees {visible_gpus} CUDA devices"
        )
    if not args.smoke and not torch.cuda.is_bf16_supported():
        raise RuntimeError("the formal baseline requires CUDA BF16 support")

    _, train_records = _validate_training_manifest(train_source)
    source_registry = _validate_fixed_eval_manifests(coco_root, cross_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    train_snapshot, snapshot_root, snapshot_registry = _prepare_snapshots(
        run_dir,
        train_source,
        coco_root,
        cross_root,
        args.smoke,
        args.smoke_manifest_records,
    )
    spec = _effective_spec(args, run_dir, args.num_gpus, source_registry, train_source, model_path)
    spec["snapshot_manifest_registry"] = snapshot_registry
    run_spec_path = run_dir / "run_spec.json"
    if run_spec_path.is_file():
        existing = read_json(run_spec_path)
        if existing != spec:
            raise ValueError("run configuration differs from this run directory; use a new directory")
    else:
        nonempty = any(run_dir.iterdir())
        if nonempty:
            allowed = {"manifests", "pipeline.lock", "status.json"}
            unexpected = [path.name for path in run_dir.iterdir() if path.name not in allowed]
            if unexpected:
                raise ValueError(f"run directory is non-empty and has no matching run_spec.json: {run_dir}")
        write_json(run_spec_path, spec)
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        write_json(config_path, {**spec, "created_at": now(), "training": {**spec["training"], "total_optimizer_steps": None}})

    grpo_dir = run_dir / "grpo"
    logs_dir = run_dir / "logs"
    grpo_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    env = _child_env(ROOT)
    eval_interval = args.smoke_eval_interval if args.smoke else 500

    state = _training_state(grpo_dir)
    if state:
        _set_total_steps(run_dir, state)

    while not _epoch_complete(state):
        latest = _latest_checkpoint(grpo_dir)
        current_step = int(state.get("global_step", _checkpoint_step(latest))) if state else _checkpoint_step(latest)
        if current_step and current_step % eval_interval == 0:
            if latest is None or _checkpoint_step(latest) != current_step:
                raise RuntimeError(f"no checkpoint at scheduled evaluation step {current_step}")
            _run_evaluation(
                args,
                run_dir=run_dir,
                snapshot_root=snapshot_root,
                checkpoint=latest,
                step=current_step,
                num_gpus=args.num_gpus,
                env=env,
            )

        target_step = ((current_step // eval_interval) + 1) * eval_interval
        command = _training_command(
            args,
            run_dir=run_dir,
            train_snapshot=train_snapshot,
            target_step=target_step,
            resume=latest,
            num_gpus=args.num_gpus,
        )
        _run_child(
            command,
            cwd=ROOT,
            env=env,
            log_path=logs_dir / f"train-to-{target_step:06d}.log",
            run_dir=run_dir,
            stage=f"train_to_{target_step}",
        )
        state = _training_state(grpo_dir)
        if not state:
            raise RuntimeError("Trainer exited without writing trainer_state.json")
        _set_total_steps(run_dir, state)
        current_step = int(state.get("global_step", 0))
        latest = _latest_checkpoint(grpo_dir)
        if current_step % eval_interval == 0 and current_step > 0:
            if latest is None or _checkpoint_step(latest) != current_step:
                raise RuntimeError(f"missing final checkpoint at step {current_step}")
            _run_evaluation(
                args,
                run_dir=run_dir,
                snapshot_root=snapshot_root,
                checkpoint=latest,
                step=current_step,
                num_gpus=args.num_gpus,
                env=env,
            )
        if not _epoch_complete(state) and current_step <= _checkpoint_step(latest):
            # TrainerState may be one checkpoint ahead after interruption, but
            # it must never make the orchestration loop spin without progress.
            if current_step >= target_step:
                continue
            raise RuntimeError(f"training did not advance to step {target_step}")

    state = _training_state(grpo_dir) or state
    final_step = int(state.get("global_step", 0))
    _set_total_steps(run_dir, state)
    latest = _latest_checkpoint(grpo_dir)
    if final_step < 1 or latest is None or _checkpoint_step(latest) != final_step:
        raise RuntimeError("the completed epoch is missing its final resumable checkpoint")
    _run_evaluation(
        args,
        run_dir=run_dir,
        snapshot_root=snapshot_root,
        checkpoint=latest,
        step=final_step,
        num_gpus=args.num_gpus,
        env=env,
    )
    _write_overall_summary(run_dir)
    write_json(run_dir / "completion.json", {
        "state": "completed",
        "final_optimizer_step": final_step,
        "total_optimizer_steps": int(state.get("max_steps", final_step)),
        "checkpoint": str(latest.resolve()),
        "completed_at": now(),
    })
    _write_status(run_dir, state="completed", stage="all", final_optimizer_step=final_step)
    print(f"Completed 1-epoch ICL→GRPO baseline: {run_dir}", flush=True)


def main() -> None:
    args = parse_args()
    run_dir = (args.run_dir.expanduser().resolve() if args.run_dir else _default_run_dir(args.smoke))
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}", flush=True)
    lock_path = run_dir / "pipeline.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another baseline process is using {run_dir}") from error
        try:
            execute(args, run_dir)
        except Exception as error:
            status_path = run_dir / "status.json"
            previous = read_json(status_path) if status_path.is_file() else {}
            write_json(run_dir / "status.json", {
                **previous,
                "state": "failed",
                "error": str(error),
                "updated_at": now(),
            })
            raise


if __name__ == "__main__":
    main()
