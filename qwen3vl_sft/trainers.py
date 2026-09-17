"""Small Trainer extensions; protocol-specific evaluation remains shared."""

from __future__ import annotations

import json
import logging
import math
import subprocess
import time
from pathlib import Path

import torch
from transformers import Trainer
from transformers.trainer_callback import PrinterCallback, ProgressCallback, TrainerCallback
from transformers.trainer_utils import speed_metrics

from .evaluation.generation import evaluate_loaded_model, load_episodes, result_payload
from .evaluation.metrics import trainer_metrics


logger = logging.getLogger(__name__)


SWANLAB_TRAIN_KEYS = ("loss", "grad_norm", "learning_rate")
SWANLAB_EVAL_KEYS = tuple(
    f"eval_coco_{shot}shot_{metric}"
    for shot in (0, 1, 2, 4)
    for metric in ("f1", "map")
)
SWANLAB_KEYS = frozenset((*SWANLAB_TRAIN_KEYS, *SWANLAB_EVAL_KEYS))


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


def _report_includes_swanlab(report_to) -> bool:
    values = [report_to] if isinstance(report_to, str) else report_to
    return any(str(item).lower() == "swanlab" for item in (values or []))


def format_iter_epoch(global_step: int, epoch: float | None, total_epochs: float) -> str:
    """Return the MMDetection-style global-iteration/epoch display."""
    epoch_total = max(1, math.ceil(total_epochs))
    completed_epochs = math.floor(epoch or 0.0)
    epoch_index = min(epoch_total, max(1, completed_epochs + 1))
    return f"{global_step}[{epoch_index}/{epoch_total}]"


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
        self.swanlab_enabled = swanlab_enabled

    def on_train_begin(self, args, state, control, **kwargs):
        del args, kwargs
        now = time.monotonic()
        self.started_at = now
        self.last_log_at = now
        self.last_log_step = state.global_step
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
        steps_per_epoch = max(1, math.ceil(total_steps / max(1, math.ceil(total_epochs))))
        epoch_number = min(
            max(1, math.floor(state.epoch or 0.0) + 1),
            max(1, math.ceil(total_epochs)),
        )
        iter_in_epoch = state.global_step - (epoch_number - 1) * steps_per_epoch
        iter_in_epoch = min(max(iter_in_epoch, 1), steps_per_epoch)
        telemetry: dict[str, float | str] = {
            "progress/iter": float(state.global_step),
            "progress/max_iters": float(total_steps),
            "progress/fraction": state.global_step / total_steps,
            "progress/epoch": float(epoch_number),
            "progress/iter_in_epoch": float(iter_in_epoch),
            "progress/iters_per_epoch": float(steps_per_epoch),
            "throughput/seconds_per_step": seconds_per_step,
            "throughput/steps_per_second": steps_per_second,
            "throughput/samples_per_second": effective_batch_size * steps_per_second,
            "throughput/effective_global_batch_size": float(effective_batch_size),
            "throughput/dataloader_workers": float(args.dataloader_num_workers),
            "throughput/elapsed_hours": total_elapsed / 3600.0,
            "throughput/eta_hours": eta_seconds / 3600.0,
        }
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
            "Iter %s | epoch_iter=%d/%d | loss=%s | lr=%s | "
            "time=%.3fs | samples/s=%.2f | eta=%.2fh",
            epoch_label,
            iter_in_epoch,
            steps_per_epoch,
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
        generation_eval_episodes: str,
        generation_eval_processor,
        generation_eval_batch_size: int,
        generation_eval_min_pixels: int,
        generation_eval_max_pixels: int,
        generation_eval_max_new_tokens: int,
        generation_eval_model_path: str,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.generation_eval_episodes = str(Path(generation_eval_episodes).resolve())
        self.generation_eval_metadata, self.generation_eval_records = load_episodes(
            self.generation_eval_episodes
        )
        self.generation_eval_processor = generation_eval_processor
        self.generation_eval_batch_size = generation_eval_batch_size
        self.generation_eval_min_pixels = generation_eval_min_pixels
        self.generation_eval_max_pixels = generation_eval_max_pixels
        self.generation_eval_max_new_tokens = generation_eval_max_new_tokens
        self.generation_eval_model_path = generation_eval_model_path

    @staticmethod
    def _distributed() -> bool:
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        del eval_dataset, ignore_keys, metric_key_prefix
        if self._distributed():
            torch.distributed.barrier()
        rank_zero = not self._distributed() or torch.distributed.get_rank() == 0
        payload = {"metrics": None, "result": None, "error": None}
        if rank_zero:
            try:
                model = self.accelerator.unwrap_model(self.model)
                device = next(model.parameters()).device
                original_cache = getattr(model.config, "use_cache", None)
                if original_cache is not None:
                    model.config.use_cache = True
                started = time.monotonic()
                try:
                    result = evaluate_loaded_model(
                        self.generation_eval_records,
                        model,
                        self.generation_eval_processor,
                        device,
                        batch_size=self.generation_eval_batch_size,
                        min_pixels=self.generation_eval_min_pixels,
                        max_pixels=self.generation_eval_max_pixels,
                        max_new_tokens=self.generation_eval_max_new_tokens,
                        coco_annotations_path=self.generation_eval_metadata.get(
                            "query_annotations"
                        ),
                    )
                finally:
                    if original_cache is not None:
                        model.config.use_cache = original_cache
                payload["metrics"] = trainer_metrics(result)
                payload["result"] = result_payload(
                    result,
                    model_path=self.generation_eval_model_path,
                    adapter_path=None,
                    episodes_path=self.generation_eval_episodes,
                    device=str(device),
                    batch_size=self.generation_eval_batch_size,
                    min_pixels=self.generation_eval_min_pixels,
                    max_pixels=self.generation_eval_max_pixels,
                    max_new_tokens=self.generation_eval_max_new_tokens,
                    coco_annotations_path=self.generation_eval_metadata.get(
                        "query_annotations"
                    ),
                    runtime_seconds=time.monotonic() - started,
                )
                output_dir = Path(self.args.output_dir) / "generation_eval"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"step-{self.state.global_step}.json"
                output_path.write_text(
                    json.dumps(payload["result"], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as error:
                payload["error"] = repr(error)
        if self._distributed():
            # Only scalar metrics and the error cross process boundaries. The
            # full prediction audit is already persisted by rank zero and can
            # be large for a 500-image manifest.
            values = [
                {
                    "metrics": payload["metrics"],
                    "error": payload["error"],
                }
                if rank_zero
                else None
            ]
            torch.distributed.broadcast_object_list(values, src=0)
            payload = values[0]
            torch.distributed.barrier()
        if payload["error"]:
            raise RuntimeError(f"generation evaluation failed: {payload['error']}")
        self.log(payload["metrics"])
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, payload["metrics"]
        )
        return payload["metrics"]
