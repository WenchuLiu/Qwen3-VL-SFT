"""Trainer telemetry and generation-evaluation integration."""

from __future__ import annotations

import logging
import math
import subprocess
import time
from pathlib import Path

import torch
from transformers import Trainer
from transformers.trainer_callback import PrinterCallback, ProgressCallback, TrainerCallback
from transformers.trainer_utils import speed_metrics

from ..evaluation.coco.generation import generate_responses, load_episodes
from ..evaluation.coco.metrics import evaluate_episode_predictions, trainer_metrics

logger = logging.getLogger(__name__)


SWANLAB_TRAIN_KEYS = ("loss", "grad_norm", "learning_rate")
SWANLAB_EVAL_KEYS = tuple(
    f"eval_coco_{shot}shot_{metric}"
    for shot in (0, 1, 2, 4)
    for metric in ("f1", "map")
)
SWANLAB_KEYS = frozenset((*SWANLAB_TRAIN_KEYS, *SWANLAB_EVAL_KEYS))


def _merge_ranked_responses(payloads: list[dict], total: int) -> list[str]:
    """Restore strided per-rank generation responses to manifest order."""
    world_size = len(payloads)
    combined: list[str | None] = [None] * total
    seen_ranks: set[int] = set()
    errors: list[str] = []
    for payload in payloads:
        rank = int(payload["rank"])
        if rank < 0 or rank >= world_size or rank in seen_ranks:
            raise RuntimeError(f"invalid or duplicate generation-eval rank: {rank}")
        seen_ranks.add(rank)
        if payload.get("error"):
            errors.append(f"rank {rank}: {payload['error']}")
            continue
        responses = list(payload.get("responses") or [])
        indices = list(range(rank, total, world_size))
        if len(responses) != len(indices):
            raise RuntimeError(
                f"rank {rank} returned {len(responses)} responses for "
                f"{len(indices)} evaluation records"
            )
        for index, response in zip(indices, responses):
            combined[index] = response
    if errors:
        raise RuntimeError("; ".join(errors))
    if seen_ranks != set(range(world_size)) or any(value is None for value in combined):
        raise RuntimeError("generation evaluation returned incomplete rank responses")
    return [str(value) for value in combined]


def _swanlab_metrics(logs: dict) -> dict[str, float]:
    """Select and normalize the only metrics that SFT uploads to SwanLab."""
    metrics: dict[str, float] = {}
    for key in SWANLAB_KEYS:
        if key not in logs:
            continue
        try:
            metrics[key] = float(logs[key])
        except (TypeError, ValueError):
            continue
    return metrics


def _format_eval_metrics(logs: dict) -> str:
    """Format evaluation metrics for the rank-zero terminal log."""
    formatted: list[str] = []
    for key in sorted(logs):
        if not key.startswith("eval_"):
            continue
        try:
            value = f"{float(logs[key]):.4f}"
        except (TypeError, ValueError):
            value = str(logs[key])
        formatted.append(f"{key}={value}")
    return " | ".join(formatted)


def _report_includes_swanlab(report_to) -> bool:
    values = [report_to] if isinstance(report_to, str) else report_to
    return any(str(item).lower() == "swanlab" for item in (values or []))


def _epoch_index(epoch: float | None, total_epochs: float) -> int:
    """Return the active epoch, keeping an exact boundary on the completed epoch."""
    epoch_total = max(1, math.ceil(total_epochs))
    epoch_value = max(float(epoch or 0.0), 0.0)
    rounded_epoch = round(epoch_value)
    if epoch_value > 0 and math.isclose(
        epoch_value,
        rounded_epoch,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        epoch_index = rounded_epoch
    else:
        epoch_index = math.ceil(epoch_value)
    return min(epoch_total, max(1, epoch_index))


def format_iter_epoch(global_step: int, epoch: float | None, total_epochs: float) -> str:
    """Return the optimizer-update/epoch display without crossing boundaries early."""
    epoch_total = max(1, math.ceil(total_epochs))
    return f"{global_step}[{_epoch_index(epoch, total_epochs)}/{epoch_total}]"


def _nvidia_smi_metrics() -> dict[str, float]:
    """Read lightweight per-GPU utilisation and memory telemetry when available."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    metrics: dict[str, float] = {}
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            continue
        try:
            index, used_mib, total_mib, utilization, temperature = map(float, fields)
        except ValueError:
            continue
        prefix = f"gpu/{int(index)}"
        metrics[f"{prefix}_memory_used_gib"] = used_mib / 1024.0
        metrics[f"{prefix}_memory_total_gib"] = total_mib / 1024.0
        metrics[f"{prefix}_memory_utilization"] = used_mib / total_mib if total_mib else 0.0
        metrics[f"{prefix}_utilization_percent"] = utilization
        metrics[f"{prefix}_temperature_celsius"] = temperature
    return metrics


class TrainingTelemetryCallback(TrainerCallback):
    """Print rich progress while uploading only the selected SFT metrics."""

    def __init__(self, swanlab_enabled: bool | None = None) -> None:
        self.started_at: float | None = None
        self.last_log_at: float | None = None
        self.last_log_step = 0
        self.micro_batches_per_epoch: int | None = None
        self.swanlab_enabled = swanlab_enabled

    def on_train_begin(self, args, state, control, **kwargs):
        del args
        now = time.monotonic()
        self.started_at = now
        self.last_log_at = now
        self.last_log_step = state.global_step
        train_dataloader = kwargs.get("train_dataloader")
        try:
            micro_batches_per_epoch = len(train_dataloader)
        except (TypeError, AttributeError):
            micro_batches_per_epoch = 0
        self.micro_batches_per_epoch = (
            int(micro_batches_per_epoch) if micro_batches_per_epoch > 0 else None
        )
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        del kwargs
        if not state.is_world_process_zero or not logs or state.global_step < 1:
            return control
        selected_metrics = _swanlab_metrics(logs)
        swanlab_enabled = (
            self.swanlab_enabled
            if self.swanlab_enabled is not None
            else _report_includes_swanlab(getattr(args, "report_to", []))
        )
        if not any(key in logs for key in SWANLAB_TRAIN_KEYS):
            eval_metrics = _format_eval_metrics(logs)
            if eval_metrics:
                logger.info("Eval @ step %d | %s", state.global_step, eval_metrics)
            if swanlab_enabled and selected_metrics:
                self._log_swanlab(selected_metrics, state.global_step)
            return control
        # Generation evaluation can take many minutes. Do not fold that wall
        # time into the training-step throughput reported at the next log.
        now = time.monotonic()
        previous_time = self.last_log_at or now
        elapsed = max(now - previous_time, 1e-9)
        step_delta = max(state.global_step - self.last_log_step, 1)
        seconds_per_step = elapsed / step_delta
        steps_per_second = step_delta / elapsed
        world_size = max(int(getattr(args, "world_size", 1)), 1)
        effective_batch_size = (
            int(args.per_device_train_batch_size)
            * world_size
            * int(args.gradient_accumulation_steps)
        )
        total_steps = max(int(state.max_steps), state.global_step)
        total_elapsed = max(now - (self.started_at or now), 0.0)
        remaining_steps = max(total_steps - state.global_step, 0)
        eta_seconds = seconds_per_step * remaining_steps
        total_epochs = float(args.num_train_epochs)
        epoch_label = format_iter_epoch(state.global_step, state.epoch, total_epochs)
        if self.micro_batches_per_epoch is not None:
            steps_per_epoch = max(
                1,
                math.ceil(
                    self.micro_batches_per_epoch
                    / max(1, int(args.gradient_accumulation_steps))
                ),
            )
        else:
            steps_per_epoch = max(
                1,
                math.ceil(total_steps / max(1, math.ceil(total_epochs))),
            )
        epoch_number = _epoch_index(state.epoch, total_epochs)
        update_in_epoch = state.global_step - (epoch_number - 1) * steps_per_epoch
        update_in_epoch = min(max(update_in_epoch, 1), steps_per_epoch)
        telemetry: dict[str, float | str] = {
            "progress/iter": float(state.global_step),
            "progress/optimizer_step": float(state.global_step),
            "progress/max_iters": float(total_steps),
            "progress/fraction": state.global_step / total_steps,
            "progress/epoch": float(epoch_number),
            "progress/iter_in_epoch": float(update_in_epoch),
            "progress/iters_per_epoch": float(steps_per_epoch),
            "progress/optimizer_update_in_epoch": float(update_in_epoch),
            "progress/optimizer_updates_per_epoch": float(steps_per_epoch),
            "throughput/seconds_per_step": seconds_per_step,
            "throughput/steps_per_second": steps_per_second,
            "throughput/samples_per_second": effective_batch_size * steps_per_second,
            "throughput/effective_global_batch_size": float(effective_batch_size),
            "throughput/dataloader_workers": float(args.dataloader_num_workers),
            "throughput/elapsed_hours": total_elapsed / 3600.0,
            "throughput/eta_hours": eta_seconds / 3600.0,
        }
        micro_batch_progress = "n/a"
        if self.micro_batches_per_epoch is not None:
            micro_batch_in_epoch = min(
                update_in_epoch * max(1, int(args.gradient_accumulation_steps)),
                self.micro_batches_per_epoch,
            )
            telemetry["progress/micro_batch_in_epoch"] = float(micro_batch_in_epoch)
            telemetry["progress/micro_batches_per_epoch"] = float(
                self.micro_batches_per_epoch
            )
            micro_batch_progress = (
                f"{micro_batch_in_epoch}/{self.micro_batches_per_epoch}"
            )
        if "loss" in logs:
            telemetry["stability/loss_is_finite"] = float(math.isfinite(float(logs["loss"])))
        if "grad_norm" in logs:
            telemetry["stability/grad_norm_is_finite"] = float(
                math.isfinite(float(logs["grad_norm"]))
            )
            telemetry["stability/grad_norm"] = float(logs["grad_norm"])
        if torch.cuda.is_available():
            telemetry["stability/cuda_allocated_gib"] = torch.cuda.memory_allocated() / 2**30
            telemetry["stability/cuda_reserved_gib"] = torch.cuda.memory_reserved() / 2**30
            telemetry["stability/cuda_peak_allocated_gib"] = (
                torch.cuda.max_memory_allocated() / 2**30
            )
        gpu_metrics = _nvidia_smi_metrics()
        telemetry.update(gpu_metrics)
        gpu_utilizations = [
            value for key, value in gpu_metrics.items() if key.endswith("_utilization_percent")
        ]
        memory_utilizations = [
            value for key, value in gpu_metrics.items() if key.endswith("_memory_utilization")
        ]
        if gpu_utilizations:
            telemetry["bottleneck/mean_gpu_utilization_percent"] = sum(gpu_utilizations) / len(
                gpu_utilizations
            )
            telemetry["bottleneck/min_gpu_utilization_percent"] = min(gpu_utilizations)
        if memory_utilizations:
            telemetry["memory/mean_gpu_utilization"] = sum(memory_utilizations) / len(
                memory_utilizations
            )
        logger.debug("Training telemetry: %s", telemetry)
        logger.info(
            "Optimizer step %s | epoch_update=%d/%d | micro_batch=%s | loss=%s | lr=%s | "
            "time=%.3fs | samples/s=%.2f | eta=%.2fh",
            epoch_label,
            update_in_epoch,
            steps_per_epoch,
            micro_batch_progress,
            f"{float(logs['loss']):.4f}" if "loss" in logs else "n/a",
            f"{float(logs['learning_rate']):.3e}" if "learning_rate" in logs else "n/a",
            seconds_per_step,
            effective_batch_size * steps_per_second,
            eta_seconds / 3600.0,
        )
        if swanlab_enabled and selected_metrics:
            self._log_swanlab(selected_metrics, state.global_step)
        self.last_log_at = now
        self.last_log_step = state.global_step
        return control

    @staticmethod
    def _log_swanlab(metrics: dict[str, float], step: int) -> None:
        try:
            import swanlab

            swanlab.log(metrics, step=step)
        except Exception as error:  # telemetry must never interrupt training
            logger.warning("SwanLab metric logging failed: %s", error)


class TrainingTelemetryTrainer(Trainer):
    """Trainer with rich rank-zero observability and MMDetection-style logs."""

    def __init__(self, *args, swanlab_enabled: bool | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the fractional ``epoch`` progress bar with explicit
        # ``Iter global_step[current_epoch/total_epochs]`` logging.
        self.remove_callback(ProgressCallback)
        self.remove_callback(PrinterCallback)
        self.add_callback(TrainingTelemetryCallback(swanlab_enabled=swanlab_enabled))

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Suppress fractional Trainer epochs; terminal telemetry shows iter[epoch]."""
        values = dict(logs)
        if self.args.include_num_input_tokens_seen != "no":
            values["num_input_tokens_seen"] = self.state.num_input_tokens_seen
            if start_time is not None:
                values.update(
                    speed_metrics(
                        "train", start_time, num_tokens=self.state.num_input_tokens_seen
                    )
                )
        output = {**values, "step": self.state.global_step}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(
            self.args, self.state, self.control, values
        )


class GenerationEvalTrainer(TrainingTelemetryTrainer):
    """Trainer whose ``evaluate`` runs fixed-episode detection metrics."""

    def __init__(
        self,
        *args,
        generation_eval_episodes: list[str] | str,
        generation_eval_processor,
        generation_eval_batch_size: int,
        generation_eval_min_pixels: int,
        generation_eval_max_pixels: int,
        generation_eval_max_new_tokens: int,
        generation_eval_media_root: str | Path | None = None,
        generation_eval_visual_enhancement: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        episode_paths = (
            [generation_eval_episodes]
            if isinstance(generation_eval_episodes, str)
            else list(generation_eval_episodes)
        )
        tasks_by_shot: dict[int, dict] = {}
        for episode_path in episode_paths:
            resolved_path = str(Path(episode_path).resolve())
            metadata, records = load_episodes(
                resolved_path,
                media_root=generation_eval_media_root,
            )
            records_by_shot: dict[int, list[dict]] = {}
            for record in records:
                shot = int(record["num_shots"])
                records_by_shot.setdefault(shot, []).append(record)
            for shot, shot_records in records_by_shot.items():
                if shot in tasks_by_shot:
                    raise ValueError(
                        f"generation evaluation shot {shot} appears in multiple manifests"
                    )
                tasks_by_shot[shot] = {
                    "shot": shot,
                    "records": shot_records,
                    "metadata": metadata,
                }
        self.generation_eval_tasks = [
            tasks_by_shot[shot] for shot in sorted(tasks_by_shot)
        ]
        self.generation_eval_processor = generation_eval_processor
        self.generation_eval_batch_size = generation_eval_batch_size
        self.generation_eval_min_pixels = generation_eval_min_pixels
        self.generation_eval_max_pixels = generation_eval_max_pixels
        self.generation_eval_max_new_tokens = generation_eval_max_new_tokens
        self.generation_eval_visual_enhancement = generation_eval_visual_enhancement

    @staticmethod
    def _distributed() -> bool:
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        del eval_dataset, ignore_keys, metric_key_prefix
        distributed = self._distributed()
        if distributed:
            torch.distributed.barrier()
        rank = torch.distributed.get_rank() if distributed else 0
        world_size = torch.distributed.get_world_size() if distributed else 1
        rank_zero = rank == 0
        model = self.accelerator.unwrap_model(self.model)
        device = next(model.parameters()).device
        original_cache = getattr(model.config, "use_cache", None)
        if original_cache is not None:
            model.config.use_cache = True
        combined_metrics: dict[str, float] = {}
        try:
            for task in self.generation_eval_tasks:
                shot = task["shot"]
                records = task["records"]
                if rank_zero:
                    logger.info(
                        "Starting %d-shot generation eval @ step %d | episodes=%d | ranks=%d",
                        shot,
                        self.state.global_step,
                        len(records),
                        world_size,
                    )
                local_records = records[rank::world_size]
                local_payload = {"rank": rank, "responses": None, "error": None}
                try:
                    local_payload["responses"] = generate_responses(
                        local_records,
                        model,
                        self.generation_eval_processor,
                        device,
                        batch_size=self.generation_eval_batch_size,
                        min_pixels=self.generation_eval_min_pixels,
                        max_pixels=self.generation_eval_max_pixels,
                        max_new_tokens=self.generation_eval_max_new_tokens,
                        visual_enhancement=self.generation_eval_visual_enhancement,
                    )
                except Exception as error:
                    local_payload["error"] = repr(error)

                if distributed:
                    gathered_payloads = [None] * world_size
                    torch.distributed.all_gather_object(gathered_payloads, local_payload)
                else:
                    gathered_payloads = [local_payload]

                task_payload = None
                if rank_zero:
                    try:
                        responses = _merge_ranked_responses(
                            gathered_payloads,
                            len(records),
                        )
                        result = evaluate_episode_predictions(
                            records,
                            responses,
                            coco_annotations_path=task["metadata"].get(
                                "query_annotations"
                            ),
                        )
                        task_payload = {
                            "metrics": trainer_metrics(result),
                            "error": None,
                        }
                    except Exception as error:
                        task_payload = {"metrics": None, "error": repr(error)}
                if distributed:
                    values = [task_payload if rank_zero else None]
                    torch.distributed.broadcast_object_list(values, src=0)
                    task_payload = values[0]
                if task_payload["error"]:
                    raise RuntimeError(
                        f"{shot}-shot generation evaluation failed: "
                        f"{task_payload['error']}"
                    )
                shot_metrics = task_payload["metrics"]
                combined_metrics.update(shot_metrics)
                self.log(shot_metrics)
        finally:
            if original_cache is not None:
                model.config.use_cache = original_cache

        self.control = self.callback_handler.on_evaluate(
            self.args,
            self.state,
            self.control,
            combined_metrics,
        )
        return combined_metrics
