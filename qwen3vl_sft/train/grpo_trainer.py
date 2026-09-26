"""A dependency-light multimodal GRPO trainer for Qwen3-VL.

TRL's GRPO trainer is designed around text prompts and leaves multimodal
preprocessing to model-specific subclasses.  This trainer keeps the same
group-relative objective while using the repository's Qwen3-VL processor and
RoPE implementation directly.
"""

from __future__ import annotations

import copy
import inspect
import math
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Callable, Mapping, Sequence

import torch
from torch.utils.checkpoint import checkpoint
from transformers import Trainer
from transformers.generation import GenerationMixin
from transformers.trainer_callback import TrainerCallback

from ..data.processing import _grid_values, _token_id
from ..data.rope import get_qwen3_rope_index
from .grpo_data import grpo_data_collator
from .rewards import REWARD_FUNCS_REGISTRY


RewardFunc = Callable[..., Sequence[float]]
_LOGPROB_VOCAB_CHUNK_SIZE = 2048


class _FSDPGenerationProxy(GenerationMixin):
    """Run GenerationMixin decoding through the FSDP-wrapped model forward."""

    # GenerationMixin's cache capability check reads this on ``cls`` rather
    # than the wrapped model instance. Qwen3-VL is not a stateful architecture.
    _is_stateful = False

    def __init__(self, fsdp_model, base_model) -> None:
        self.__dict__["_fsdp_model"] = fsdp_model
        self.__dict__["_base_model"] = base_model
        generation_model = (
            base_model.get_base_model()
            if callable(getattr(base_model, "get_base_model", None))
            else base_model
        )

        def forward_through_fsdp(*args, **kwargs):
            return fsdp_model(*args, **kwargs)

        # GenerationMixin inspects the forward signature to discover model
        # inputs. Keep the underlying Qwen signature while dispatching calls
        # through the FSDP wrapper so each FSDP unit can all-gather as needed.
        forward_through_fsdp.__signature__ = inspect.signature(generation_model.forward)
        self.__dict__["forward"] = forward_through_fsdp

    def __getattr__(self, name: str):
        return getattr(self.__dict__["_base_model"], name)

    def __call__(self, *args, **kwargs):
        return self.__dict__["_fsdp_model"](*args, **kwargs)


def _chunked_logsumexp(logits: torch.Tensor) -> torch.Tensor:
    """Compute log-sum-exp over the vocabulary without a full-size temporary.

    The LM head logits can be several GiB for multimodal GRPO batches.  A
    full-vocabulary ``log_softmax`` allocates another tensor of the same
    shape.  Reducing vocabulary slices and combining their log-sum-exp values
    bounds temporary memory by ``_LOGPROB_VOCAB_CHUNK_SIZE`` instead.  During
    training, checkpoint each reduction so backward recomputes one slice at a
    time rather than retaining all of the float32 intermediates.
    """
    vocab_size = logits.shape[-1]
    log_normalizer = None

    def reduce_chunk(chunk: torch.Tensor) -> torch.Tensor:
        # Accumulate in float32, matching log_softmax's numerically stable
        # behavior under bf16/fp16 model execution without upcasting all logits.
        return torch.logsumexp(chunk.float(), dim=-1)

    for start in range(0, vocab_size, _LOGPROB_VOCAB_CHUNK_SIZE):
        end = min(start + _LOGPROB_VOCAB_CHUNK_SIZE, vocab_size)
        chunk = logits[..., start:end]
        if torch.is_grad_enabled() and chunk.requires_grad:
            chunk_log_normalizer = checkpoint(
                reduce_chunk,
                chunk,
                use_reentrant=False,
            )
        else:
            chunk_log_normalizer = reduce_chunk(chunk)

        if log_normalizer is None:
            log_normalizer = chunk_log_normalizer
        else:
            log_normalizer = torch.logaddexp(
                log_normalizer,
                chunk_log_normalizer,
            )

    if log_normalizer is None:
        raise ValueError("logits must have a non-empty vocabulary dimension")
    return log_normalizer


def _grid_tensor(value: object) -> torch.Tensor | None:
    values = _grid_values(value)
    if not values:
        return None
    return torch.cat(values, dim=0)


def _move_value(value: object, device: torch.device) -> object:
    return value.to(device) if hasattr(value, "to") else value


class GRPOSwanLabCallback(TrainerCallback):
    """Forward GRPO metrics to an already initialized SwanLab run."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def on_log(self, args, state, control, logs=None, **kwargs):
        del args, kwargs
        if not self.enabled or not state.is_world_process_zero or not logs:
            return control
        try:
            import swanlab

            metrics = {}
            for key, value in logs.items():
                if key == "step" or not isinstance(value, (int, float)):
                    continue
                if math.isfinite(float(value)):
                    metrics[key] = float(value)
            if metrics:
                swanlab.log(metrics, step=state.global_step)
        except Exception:
            # Experiment logging must not stop policy optimization.
            pass
        return control


class MultimodalGRPOTrainer(Trainer):
    """Group Relative Policy Optimization with Qwen3-VL rollouts.

    A dataset item contains ``prompt`` (multimodal chat messages),
    ``target_boxes``, and optional reward metadata.  At every optimizer step
    the trainer samples ``num_generations`` completions for each prompt,
    computes the configured rewards, normalizes them inside each group, and
    optimizes the sampled-token log probabilities with a KL penalty.
    """

    def __init__(
        self,
        *,
        model,
        processor,
        reward_funcs: Sequence[RewardFunc | str],
        args,
        train_dataset=None,
        eval_dataset=None,
        reference_model=None,
        reward_weights: Sequence[float] | None = None,
        swanlab_enabled: bool = False,
        callbacks=None,
        **kwargs,
    ) -> None:
        resolved_funcs: list[RewardFunc] = []
        reward_names: list[str] = []
        for reward_func in reward_funcs:
            if isinstance(reward_func, str):
                try:
                    function = REWARD_FUNCS_REGISTRY[reward_func]
                except KeyError as error:
                    available = ", ".join(sorted(REWARD_FUNCS_REGISTRY))
                    raise ValueError(
                        f"unknown reward function {reward_func!r}; choose from {available}"
                    ) from error
                reward_names.append(reward_func)
            else:
                function = reward_func
                reward_names.append(getattr(function, "__name__", "reward"))
            resolved_funcs.append(function)
        if not resolved_funcs:
            raise ValueError("at least one reward function is required")

        num_generations = int(getattr(args, "num_generations", 0))
        if num_generations < 2:
            raise ValueError("num_generations must be at least 2 for group-relative rewards")
        max_completion_length = int(getattr(args, "max_completion_length", 0))
        if max_completion_length < 1:
            raise ValueError("max_completion_length must be positive")
        temperature = float(getattr(args, "temperature", 1.0))
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        top_p = float(getattr(args, "top_p", 1.0))
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        top_k = int(getattr(args, "top_k", 0))
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        beta = float(getattr(args, "kl_coef", 0.0))
        if beta < 0:
            raise ValueError("kl_coef must be non-negative")

        if reward_weights is None:
            reward_weights = [1.0] * len(resolved_funcs)
        if len(reward_weights) != len(resolved_funcs):
            raise ValueError("reward_weights must match reward_funcs")
        if any(float(weight) < 0 for weight in reward_weights):
            raise ValueError("reward weights must be non-negative")

        self.processor = processor
        self.reward_funcs = resolved_funcs
        self.reward_names = reward_names
        self.reward_weights = torch.tensor(reward_weights, dtype=torch.float32)
        self.num_generations = num_generations
        self.max_prompt_length = int(getattr(args, "max_prompt_length", 0))
        self.max_completion_length = max_completion_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.beta = beta
        self.iou_threshold = float(getattr(args, "iou_threshold", 0.5))
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1]")
        self._metrics = defaultdict(list)

        if self.max_prompt_length < 1:
            raise ValueError("max_prompt_length must be positive")

        # LoRA models can disable their adapter to expose the frozen base
        # policy.  Full-parameter training needs a separate frozen copy for
        # the KL term; this is the same reference-policy distinction used by
        # the Visual-RFT trainer.
        if beta > 0 and reference_model is None and not hasattr(model, "disable_adapter"):
            reference_model = copy.deepcopy(model)
        if beta > 0 and reference_model is None and not hasattr(model, "disable_adapter"):
            raise ValueError(
                "a KL reference model is required for full-parameter GRPO; "
                "use LoRA or pass reference_model"
            )
        self.ref_model = reference_model
        if self.ref_model is not None:
            self.ref_model.eval()
            for parameter in self.ref_model.parameters():
                parameter.requires_grad_(False)

        if callbacks is None:
            callbacks = []
        else:
            callbacks = list(callbacks)
        callbacks.append(GRPOSwanLabCallback(enabled=swanlab_enabled))

        if kwargs.get("data_collator") is None:
            kwargs["data_collator"] = grpo_data_collator
        args.remove_unused_columns = False
        # ``processing_class`` is used for checkpoint metadata and tokenizer
        # saving; multimodal rendering itself uses ``self.processor`` above.
        kwargs.setdefault("processing_class", processor.tokenizer)

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
            **kwargs,
        )
        self.model_accepts_loss_kwargs = False
        if self.ref_model is not None:
            self.ref_model = self.accelerator.prepare_model(
                self.ref_model, evaluation_mode=True
            )

    def _set_signature_columns_if_needed(self):
        # Rollout metadata is deliberately consumed by reward functions rather
        # than by the model's forward signature.
        if self._signature_columns is None:
            self._signature_columns = [
                "prompt",
                "target_boxes",
                "target_labels",
                "category",
                "solution",
                "id",
            ]

    def _prepare_inputs(self, inputs):
        # The default Trainer recursively moves tensors in a batch to the
        # accelerator.  GRPO batches are prompt dictionaries containing image
        # paths and are rendered below, so leave them untouched.
        return inputs

    @staticmethod
    def _model_inputs(inputs: Mapping[str, object]) -> dict[str, object]:
        allowed = {
            "input_ids",
            "attention_mask",
            "position_ids",
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "second_per_grid_ts",
            "rope_deltas",
        }
        return {key: value for key, value in inputs.items() if key in allowed and value is not None}

    def _ensure_position_ids(self, inputs: dict[str, object]) -> dict[str, object]:
        input_ids = inputs["input_ids"]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
            inputs["input_ids"] = input_ids
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            inputs["attention_mask"] = attention_mask
        if "position_ids" in inputs:
            position_ids = inputs["position_ids"]
            if position_ids.ndim == 2:
                position_ids = position_ids.unsqueeze(1)
            inputs["position_ids"] = position_ids
            return inputs

        image_grid = _grid_tensor(inputs.get("image_grid_thw"))
        video_grid = _grid_tensor(inputs.get("video_grid_thw"))
        tokenizer = self.processor.tokenizer
        position_ids, _ = get_qwen3_rope_index(
            spatial_merge_size=int(getattr(self.processor.image_processor, "merge_size", 2)),
            input_ids=input_ids,
            image_grid_thw=image_grid,
            video_grid_thw=video_grid,
            attention_mask=attention_mask,
            image_token_id=_token_id(tokenizer, "<|image_pad|>", 151655) or 151655,
            video_token_id=_token_id(tokenizer, "<|video_pad|>", 151656) or 151656,
            vision_start_token_id=_token_id(tokenizer, "<|vision_start|>", 151652) or 151652,
        )
        inputs["position_ids"] = position_ids
        return inputs

    def _render_prompt(self, sample: Mapping[str, object]) -> dict[str, object]:
        try:
            rendered = self.processor.apply_chat_template(
                sample["prompt"],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
        except TypeError:
            # A small compatibility fallback for processors that infer the
            # generation prompt from ``tokenize=True``.
            rendered = self.processor.apply_chat_template(
                sample["prompt"],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        inputs = dict(rendered)
        inputs = self._ensure_position_ids(inputs)
        if inputs["input_ids"].shape[-1] > self.max_prompt_length:
            raise ValueError(
                "a GRPO prompt exceeds max_prompt_length; increase the limit "
                "because truncating multimodal prompts can desynchronize vision tokens"
            )
        device = self.accelerator.device
        return {
            key: _move_value(value, device)
            for key, value in self._model_inputs(inputs).items()
        }

    def _generate_completions(self, model, prompt_inputs: Mapping[str, object]) -> torch.Tensor:
        """Sample one completion at a time to preserve multimodal alignment."""
        unwrapped_model = self.accelerator.unwrap_model(model)
        is_fsdp = False
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            is_fsdp = isinstance(model, FSDP)
        except ImportError:
            pass
        generation_model = (
            _FSDPGenerationProxy(model, unwrapped_model) if is_fsdp else unwrapped_model
        )
        was_training = model.training
        original_use_cache = getattr(unwrapped_model.config, "use_cache", None)
        if original_use_cache is not None:
            unwrapped_model.config.use_cache = True
        model.eval()
        sequences = []
        try:
            for _ in range(self.num_generations):
                generation_kwargs = dict(prompt_inputs)
                # Qwen3-VL's generation helper computes cache-aware RoPE
                # positions as tokens are appended. The full-sequence forward
                # below recomputes positions explicitly for policy log-probs.
                generation_kwargs.pop("position_ids", None)
                generation_kwargs.update(
                    max_new_tokens=self.max_completion_length,
                    do_sample=self.temperature > 0,
                    num_return_sequences=1,
                )
                if is_fsdp:
                    # Each rank samples independently, so keep FSDP forward
                    # collectives aligned when ranks finish at different times.
                    generation_kwargs["synced_gpus"] = True
                if self.temperature > 0:
                    generation_kwargs["temperature"] = self.temperature
                    generation_kwargs["top_p"] = self.top_p
                    if self.top_k:
                        generation_kwargs["top_k"] = self.top_k
                with torch.inference_mode():
                    generated = generation_model.generate(**generation_kwargs)
                if hasattr(generated, "sequences"):
                    generated = generated.sequences
                sequences.append(generated[0])
        finally:
            if original_use_cache is not None:
                unwrapped_model.config.use_cache = original_use_cache
            model.train(was_training)

        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processor.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0
        return torch.nn.utils.rnn.pad_sequence(
            sequences,
            batch_first=True,
            padding_value=int(pad_token_id),
        )

    def _completion_mask(self, completion_ids: torch.Tensor) -> torch.Tensor:
        eos_token_id = self.processor.tokenizer.eos_token_id
        eos_ids = {int(eos_token_id)} if isinstance(eos_token_id, int) else set(eos_token_id or [])
        pad_token_id = self.processor.tokenizer.pad_token_id
        mask = torch.ones_like(completion_ids, dtype=torch.float32)
        for row_index, row in enumerate(completion_ids):
            end = row.numel()
            for index, token in enumerate(row.tolist()):
                if token in eos_ids:
                    end = index + 1
                    break
                if pad_token_id is not None and token == int(pad_token_id):
                    end = index
                    break
            if end < row.numel():
                mask[row_index, end:] = 0.0
        return mask

    @staticmethod
    def _repeat_prompt_inputs(
        prompt_inputs: Mapping[str, object], repeats: int
    ) -> dict[str, object]:
        repeated: dict[str, object] = {}
        for key, value in prompt_inputs.items():
            if not hasattr(value, "ndim"):
                repeated[key] = value
            elif key in {"input_ids", "attention_mask"}:
                repeated[key] = value.repeat_interleave(repeats, dim=0)
            elif key == "position_ids":
                repeated[key] = value.repeat_interleave(repeats, dim=1)
            elif key in {
                "pixel_values",
                "image_grid_thw",
                "pixel_values_videos",
                "video_grid_thw",
                "second_per_grid_ts",
            }:
                repeated[key] = torch.cat([value] * repeats, dim=0)
            else:
                repeated[key] = value
        return repeated

    def _full_position_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        model_inputs: Mapping[str, object],
    ) -> torch.Tensor:
        image_grid = _grid_tensor(model_inputs.get("image_grid_thw"))
        video_grid = _grid_tensor(model_inputs.get("video_grid_thw"))
        tokenizer = self.processor.tokenizer
        position_ids, _ = get_qwen3_rope_index(
            spatial_merge_size=int(getattr(self.processor.image_processor, "merge_size", 2)),
            input_ids=input_ids,
            image_grid_thw=image_grid,
            video_grid_thw=video_grid,
            attention_mask=attention_mask,
            image_token_id=_token_id(tokenizer, "<|image_pad|>", 151655) or 151655,
            video_token_id=_token_id(tokenizer, "<|video_pad|>", 151656) or 151656,
            vision_start_token_id=_token_id(tokenizer, "<|vision_start|>", 151652) or 151652,
        )
        return position_ids

    def _get_per_token_logps(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        model_inputs: Mapping[str, object],
        prompt_length: int,
    ) -> torch.Tensor:
        kwargs = self._model_inputs(model_inputs)
        kwargs["input_ids"] = input_ids
        kwargs["attention_mask"] = attention_mask
        kwargs["position_ids"] = self._full_position_ids(input_ids, attention_mask, kwargs)
        kwargs.pop("rope_deltas", None)
        outputs = model(**kwargs, use_cache=False)
        logits = outputs.logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        target_logits = torch.gather(
            logits,
            dim=-1,
            index=target_ids.unsqueeze(-1),
        ).squeeze(-1).float()
        token_logps = target_logits - _chunked_logsumexp(logits)
        return token_logps[:, prompt_length - 1 :]

    def _compute_rewards(
        self,
        sample: Mapping[str, object],
        prompts: list[object],
        completions: list[list[dict[str, str]]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.accelerator.device
        raw_rewards = torch.zeros(
            self.num_generations,
            len(self.reward_funcs),
            dtype=torch.float32,
            device=device,
        )
        reward_kwargs = {
            key: [value] * self.num_generations
            for key, value in sample.items()
            if key not in {"prompt", "completion"}
        }
        for reward_index, reward_func in enumerate(self.reward_funcs):
            if self.reward_names[reward_index] in {
                "iou",
                "score",
                "confidence",
                "accuracy_iou",
                "accuracy_confidence",
            }:
                reward_kwargs["iou_threshold"] = self.iou_threshold
            values = reward_func(
                prompts=prompts,
                completions=completions,
                **reward_kwargs,
            )
            if isinstance(values, torch.Tensor):
                values = values.detach().to(device=device, dtype=torch.float32).flatten()
            else:
                try:
                    values = torch.tensor(list(values), dtype=torch.float32, device=device)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"reward function {self.reward_names[reward_index]!r} "
                        "did not return numbers"
                    ) from error
            if values.numel() != self.num_generations:
                raise ValueError(
                    f"reward function {self.reward_names[reward_index]!r} returned "
                    f"{values.numel()} rewards for {self.num_generations} completions"
                )
            raw_rewards[:, reward_index] = values
        weights = self.reward_weights.to(device=device)
        return raw_rewards, (raw_rewards * weights.unsqueeze(0)).sum(dim=1)

    def _sample_loss(
        self, model, sample: Mapping[str, object]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        prompt_inputs = self._render_prompt(sample)
        prompt_ids = prompt_inputs["input_ids"]
        prompt_mask = prompt_inputs["attention_mask"]
        prompt_length = prompt_ids.shape[-1]
        generated = self._generate_completions(model, prompt_inputs)
        completion_ids = generated[:, prompt_length:]
        if completion_ids.shape[-1] < 1:
            raise RuntimeError("generation returned an empty completion")
        completion_mask = self._completion_mask(completion_ids)

        texts = self.processor.batch_decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        completions = [[{"role": "assistant", "content": text}] for text in texts]
        prompts = [sample["prompt"]] * self.num_generations
        rewards_per_func, rewards = self._compute_rewards(sample, prompts, completions)

        repeated_inputs = self._repeat_prompt_inputs(prompt_inputs, self.num_generations)
        full_ids = torch.cat(
            [prompt_ids.repeat(self.num_generations, 1), completion_ids],
            dim=1,
        )
        attention_mask = torch.cat(
            [prompt_mask.repeat(self.num_generations, 1), completion_mask.long()],
            dim=1,
        )
        per_token_logps = self._get_per_token_logps(
            model,
            full_ids,
            attention_mask,
            repeated_inputs,
            prompt_length,
        )

        if self.beta > 0:
            with torch.inference_mode():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model,
                        full_ids,
                        attention_mask,
                        repeated_inputs,
                        prompt_length,
                    )
                else:
                    base_model = self.accelerator.unwrap_model(model)
                    adapter_context = (
                        base_model.disable_adapter()
                        if hasattr(base_model, "disable_adapter")
                        else nullcontext()
                    )
                    with adapter_context:
                        ref_per_token_logps = self._get_per_token_logps(
                            model,
                            full_ids,
                            attention_mask,
                            repeated_inputs,
                            prompt_length,
                        )
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (
                ref_per_token_logps - per_token_logps
            ) - 1.0
        else:
            per_token_kl = torch.zeros_like(per_token_logps)

        mean_reward = rewards.mean()
        reward_std = rewards.std(unbiased=False)
        advantages = (rewards - mean_reward) / (reward_std + 1e-4)
        ratio = torch.exp(per_token_logps - per_token_logps.detach())
        token_count = completion_mask.sum(dim=1).clamp_min(1.0)
        policy_loss = -(
            ratio * advantages.detach().unsqueeze(1) * completion_mask
        ).sum(dim=1) / token_count
        kl_loss = (per_token_kl * completion_mask).sum(dim=1) / token_count
        loss = (policy_loss + self.beta * kl_loss).mean()

        stats = {
            "reward": float(mean_reward.detach().item()),
            "reward_std": float(reward_std.detach().item()),
            "kl": float(kl_loss.detach().mean().item()),
            "completion_length": float(token_count.detach().mean().item()),
        }
        for index, name in enumerate(self.reward_names):
            stats[f"rewards/{name}"] = float(rewards_per_func[:, index].mean().detach().item())
        return loss, stats

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        if return_outputs:
            raise ValueError("MultimodalGRPOTrainer does not support returning outputs")
        if isinstance(inputs, Mapping):
            inputs = [inputs]
        if not isinstance(inputs, Sequence) or not inputs:
            raise ValueError("GRPO received an empty batch")
        losses = []
        batch_stats = defaultdict(list)
        for sample in inputs:
            loss, stats = self._sample_loss(model, sample)
            losses.append(loss)
            for key, value in stats.items():
                batch_stats[key].append(value)
        for key, values in batch_stats.items():
            self._metrics[key].append(sum(values) / len(values))
        return torch.stack(losses).mean()

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        averaged = {
            key: sum(values) / len(values)
            for key, values in self._metrics.items()
            if values
        }
        merged = {**logs, **averaged}
        try:
            super().log(merged, start_time=start_time)
        except TypeError:
            # Transformers versions before the start_time argument are still
            # useful for local smoke tests.
            super().log(merged)
        self._metrics.clear()


# Short aliases make the trainer discoverable from both names used in
# multimodal GRPO examples and this repository's package naming.
Qwen3VLGRPOTrainer = MultimodalGRPOTrainer


__all__ = ["MultimodalGRPOTrainer", "Qwen3VLGRPOTrainer"]
