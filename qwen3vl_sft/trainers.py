"""Small Trainer extensions; protocol-specific evaluation remains shared."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from transformers import Trainer

from .evaluation.generation import evaluate_loaded_model, load_episodes, result_payload
from .evaluation.metrics import trainer_metrics


class GenerationEvalTrainer(Trainer):
    """Hugging Face Trainer whose ``evaluate`` is fixed-episode generation F1."""

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
        _, self.generation_eval_records = load_episodes(self.generation_eval_episodes)
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
