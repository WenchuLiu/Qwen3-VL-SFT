"""A dependency-light multimodal GRPO trainer for Qwen3-VL.

TRL's GRPO trainer is designed around text prompts and leaves multimodal
preprocessing to model-specific subclasses.  This trainer keeps the same
group-relative objective while using the repository's Qwen3-VL processor and
RoPE implementation directly.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Mapping, Sequence

import torch
from torch.utils.checkpoint import checkpoint
from transformers import Trainer
from transformers.trainer_callback import TrainerCallback

from ..data.processing import _grid_values, _token_id
from ..data.rope import get_qwen3_rope_index
from .grpo_data import grpo_data_collator
from .rewards import REWARD_FUNCS_REGISTRY


RewardFunc = Callable[..., Sequence[float]]
_LOGPROB_VOCAB_CHUNK_SIZE = 2048


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


def _add_initial_policy_reference_adapter(model) -> tuple[str, str]:
    """Clone the active PEFT adapter so KL stays anchored to the initial policy."""
    peft_config = getattr(model, "peft_config", None)
    if not peft_config:
        raise ValueError("cannot create a GRPO reference adapter on a non-PEFT model")

    active = getattr(model, "active_adapters", None)
    if callable(active):
        active = active()
    if not active:
        active = getattr(model, "active_adapter", None)
        if callable(active):
            active = active()
    if isinstance(active, str):
        active = [active]
    if not active and len(peft_config) == 1:
        active = [next(iter(peft_config))]
    if len(active or []) != 1:
        raise ValueError(
            "GRPO KL anchoring requires exactly one active PEFT adapter; "
            "select one adapter before training"
        )

    policy_adapter = active[0]
    reference_adapter = "grpo_reference"
    if reference_adapter in peft_config:
        raise ValueError(
            f"adapter {reference_adapter!r} is reserved for the frozen GRPO reference"
        )

    from peft.utils.save_and_load import (
        get_peft_model_state_dict,
        set_peft_model_state_dict,
    )

    policy_state = get_peft_model_state_dict(model, adapter_name=policy_adapter)
    model.add_adapter(reference_adapter, copy.deepcopy(peft_config[policy_adapter]))
    set_peft_model_state_dict(
        model,
        policy_state,
        adapter_name=reference_adapter,
    )
    model.set_adapter(policy_adapter)
    return policy_adapter, reference_adapter


def _disable_dropout(model) -> None:
    """Keep rollout and policy scoring deterministic in GRPO."""
    if model is None:
        return
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    for config in getattr(model, "peft_config", {}).values():
        if hasattr(config, "lora_dropout"):
            config.lora_dropout = 0.0


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
            raise ValueError("max_completion_length must be a positive integer")
        temperature = float(getattr(args, "temperature", 1.0))
        if temperature <= 0:
            raise ValueError("temperature must be positive for GRPO sampling")
        top_p = float(getattr(args, "top_p", 1.0))
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        top_k = int(getattr(args, "top_k", 0))
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if (
            not math.isclose(temperature, 1.0, rel_tol=0.0, abs_tol=1e-8)
            or top_p != 1.0
            or top_k != 0
        ):
            raise ValueError(
                "this GRPO trainer scores the unwarped policy distribution; "
                "use temperature=1, top_p=1, and top_k=0"
            )
        clip_epsilon = float(getattr(args, "clip_epsilon", 0.2))
        if not 0.0 <= clip_epsilon < 1.0:
            raise ValueError("clip_epsilon must be in [0, 1)")
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
        self.clip_epsilon = clip_epsilon
        self.beta = beta
        self.iou_threshold = float(getattr(args, "iou_threshold", 0.5))
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1]")
        self._metrics = defaultdict(list)

        if self.max_prompt_length < 1:
            raise ValueError("max_prompt_length must be positive")

        # Freeze a copy of the initial PEFT adapter as the KL anchor. Disabling
        # the adapter would compare against the bare base model and discard the
        # SFT policy that GRPO is meant to refine.
        self.policy_adapter_name = None
        self.reference_adapter_name = None
        if beta > 0 and reference_model is None and getattr(model, "peft_config", None):
            (
                self.policy_adapter_name,
                self.reference_adapter_name,
            ) = _add_initial_policy_reference_adapter(model)

        _disable_dropout(model)
        _disable_dropout(reference_model)
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

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        target = model if model is not None else self.model
        checkpoint = Path(resume_from_checkpoint)
        # Transformers 4.57 loads adapter subdirectories instead of the root
        # adapter when both are present. GRPO saves the policy at the root and
        # the frozen SFT reference in grpo_reference/, so restore both.
        if (
            self.reference_adapter_name is not None
            and (checkpoint / "adapter_config.json").is_file()
            and (checkpoint / self.reference_adapter_name / "adapter_config.json").is_file()
        ):
            target.load_adapter(
                str(checkpoint), self.policy_adapter_name, is_trainable=True
            )
            target.set_adapter(self.policy_adapter_name)

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
        # DDP generation uses the local replica without gradient synchronization;
        # policy scoring still goes through the DDP wrapper.
        generation_model = unwrapped_model
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
                    do_sample=True,
                    num_return_sequences=1,
                    # Override checkpoint generation defaults so the sampled
                    # distribution matches the raw token log-probabilities.
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                )
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
        completion_length = input_ids.shape[-1] - prompt_length
        if not 0 <= completion_length <= self.max_completion_length:
            raise ValueError(
                "completion length must be between 0 and the configured maximum "
                f"({self.max_completion_length}); got {completion_length}"
            )

        kwargs = self._model_inputs(model_inputs)
        kwargs["input_ids"] = input_ids
        kwargs["attention_mask"] = attention_mask
        kwargs["position_ids"] = self._full_position_ids(input_ids, attention_mask, kwargs)
        kwargs.pop("rope_deltas", None)

        # The final prompt logit predicts the first completion token. Keep
        # completion_length + 1 positions, then discard the unused final logit.
        outputs = model(
            **kwargs,
            use_cache=False,
            logits_to_keep=completion_length + 1,
        )
        logits = outputs.logits[:, :-1, :]
        target_ids = input_ids[:, prompt_length:]
        target_logits = torch.gather(
            logits,
            dim=-1,
            index=target_ids.unsqueeze(-1),
        ).squeeze(-1).float()
        return target_logits - _chunked_logsumexp(logits)

    def _disable_accelerate_fp32_output_conversion(self, model) -> None:
        """Avoid materializing a full FP32 copy of BF16 vocabulary logits.

        Trainer's autocast context still controls the forward dtype. The
        Accelerate output wrapper otherwise converts the complete selected
        logits tensor to FP32 before this trainer can reduce it by vocab chunks.
        """
        accelerator = getattr(self, "accelerator", None)
        if accelerator is None:
            return
        try:
            accelerator.unwrap_model(model, keep_fp32_wrapper=False)
        except TypeError:
            # Older Accelerate versions do not expose keep_fp32_wrapper.
            unwrapped_model = accelerator.unwrap_model(model)
            original_forward = getattr(unwrapped_model, "_original_forward", None)
            if original_forward is not None:
                unwrapped_model.forward = original_forward
                delattr(unwrapped_model, "_original_forward")

    def _grpo_autocast_context(self):
        """Keep Accelerate mixed precision after removing its output upcast."""
        accelerator = getattr(self, "accelerator", None)
        autocast = getattr(accelerator, "autocast", None)
        if callable(autocast):
            return autocast()
        context_factory = getattr(self, "compute_loss_context_manager", None)
        return context_factory() if callable(context_factory) else nullcontext()

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
                "score",
                "confidence",
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
        self,
        model,
        sample: Mapping[str, object],
        *,
        backward_callback: Callable[[torch.Tensor], None] | None = None,
        backward_context: Callable[[bool], Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Score a GRPO group, optionally backpropagating one completion at a time.

        Trainer's regular compute_loss path keeps every completion graph alive
        until the whole group loss is returned. The training path passes a
        backward callback so each completion graph is released immediately.
        """
        if backward_callback is None and backward_context is not None:
            raise ValueError("backward_context requires backward_callback")

        # Remove Accelerate's automatic full-output upcast. Its autocast context
        # is restored explicitly around all policy, reference, and rollout forwards.
        self._disable_accelerate_fp32_output_conversion(model)
        if self.beta > 0 and self.ref_model is not None:
            self._disable_accelerate_fp32_output_conversion(self.ref_model)

        if backward_callback is None:
            with self._grpo_autocast_context():
                return self._sample_loss_batched(model, sample)
        if backward_context is None:
            backward_context = lambda _is_last: nullcontext()
        return self._sample_loss_streamed(
            model,
            sample,
            backward_callback=backward_callback,
            backward_context=backward_context,
        )

    @contextmanager
    def _gradient_checkpointing_layer_context(self, model):
        """Enable checkpointed transformer layers while keeping eval semantics.

        GRPO rollout and behavior-policy scoring run in eval mode so dropout is
        disabled. That also disables Transformers' layer-level checkpointing,
        which checks each layer's ``training`` flag. During the gradient-bearing
        policy pass, temporarily enable only checkpointable layers; their child
        dropout modules and the rest of the model remain in eval mode.
        """
        changed = []
        for module in model.modules():
            checkpoint_fn = getattr(module, "_gradient_checkpointing_func", None)
            if getattr(module, "gradient_checkpointing", False) and callable(checkpoint_fn):
                changed.append((module, module.training))
                module.training = True
        try:
            yield
        finally:
            for module, was_training in changed:
                module.training = was_training

    def _sample_loss_streamed(
        self,
        model,
        sample: Mapping[str, object],
        *,
        backward_callback: Callable[[torch.Tensor], None],
        backward_context: Callable[[bool], Any],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Backpropagate each completion before constructing the next graph."""
        was_training = model.training
        model.eval()
        try:
            prompt_inputs = self._render_prompt(sample)
            prompt_ids = prompt_inputs["input_ids"]
            prompt_mask = prompt_inputs["attention_mask"]
            prompt_length = prompt_ids.shape[-1]
            with self._grpo_autocast_context():
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

            mean_reward = rewards.mean()
            reward_std = rewards.std(unbiased=False)
            advantages = ((rewards - mean_reward) / (reward_std + 1e-4)).detach()

            # Keep only token IDs and masks for each candidate. The multimodal
            # prompt (especially pixel_values) is reused as-is on each forward,
            # rather than concatenated num_generations times.
            candidate_batches = []
            for index in range(self.num_generations):
                valid_length = int(completion_mask[index].sum().item())
                # Preserve a graph-connected, all-masked token for a degenerate
                # empty completion so every DDP rank still participates.
                candidate_length = max(valid_length, 1)
                candidate_completion = completion_ids[
                    index : index + 1, :candidate_length
                ]
                candidate_mask = completion_mask[index : index + 1, :candidate_length]
                candidate_input_ids = torch.cat(
                    [prompt_ids, candidate_completion],
                    dim=1,
                )
                candidate_attention_mask = torch.cat(
                    [prompt_mask, candidate_mask.long()],
                    dim=1,
                )
                candidate_batches.append(
                    (candidate_input_ids, candidate_attention_mask, candidate_mask)
                )

            # These detached log-probabilities are small [1, completion_length]
            # tensors. Compute them serially before any current-policy graph is
            # built, matching the rollout policy and reference semantics.
            old_per_token_logps = []
            with torch.no_grad():
                for candidate_input_ids, candidate_attention_mask, _ in candidate_batches:
                    with self._grpo_autocast_context():
                        old_logps = self._get_per_token_logps(
                            model,
                            candidate_input_ids,
                            candidate_attention_mask,
                            prompt_inputs,
                            prompt_length,
                        )
                    old_per_token_logps.append(old_logps.detach())

            ref_per_token_logps = None
            if self.beta > 0:
                reference_model = self.ref_model
                adapter_context = nullcontext()
                switched_reference_adapter = False
                base_model = None
                if reference_model is None:
                    base_model = self.accelerator.unwrap_model(model)
                    reference_model = model
                    if self.reference_adapter_name is not None:
                        base_model.set_adapter(self.reference_adapter_name)
                        switched_reference_adapter = True
                    else:
                        adapter_context = (
                            base_model.disable_adapter()
                            if hasattr(base_model, "disable_adapter")
                            else nullcontext()
                        )
                ref_per_token_logps = []
                try:
                    with adapter_context:
                        with torch.no_grad():
                            for (
                                candidate_input_ids,
                                candidate_attention_mask,
                                _,
                            ) in candidate_batches:
                                with self._grpo_autocast_context():
                                    ref_logps = self._get_per_token_logps(
                                        reference_model,
                                        candidate_input_ids,
                                        candidate_attention_mask,
                                        prompt_inputs,
                                        prompt_length,
                                    )
                                ref_per_token_logps.append(ref_logps.detach())
                finally:
                    if switched_reference_adapter:
                        base_model.set_adapter(self.policy_adapter_name)

            loss_total = torch.zeros((), device=rewards.device, dtype=torch.float32)
            kl_total = torch.zeros_like(loss_total)
            clipped_token_total = torch.zeros_like(loss_total)
            valid_token_total = torch.zeros_like(loss_total)
            completion_length_total = torch.zeros_like(loss_total)

            with self._gradient_checkpointing_layer_context(model):
                for index, (
                    candidate_input_ids,
                    candidate_attention_mask,
                    candidate_mask,
                ) in enumerate(candidate_batches):
                    old_logps = old_per_token_logps[index]
                    valid_tokens = candidate_mask.bool()

                    # DDP requires no_sync to enclose both forward and backward.
                    # The training_step callback selects the one final completion
                    # in the local microbatch that synchronizes accumulated grads.
                    with backward_context(index == self.num_generations - 1):
                        with self._grpo_autocast_context():
                            per_token_logps = self._get_per_token_logps(
                                model,
                                candidate_input_ids,
                                candidate_attention_mask,
                                prompt_inputs,
                                prompt_length,
                            )
                            if valid_tokens.any() and not torch.allclose(
                                per_token_logps.detach()[valid_tokens],
                                old_logps[valid_tokens],
                                atol=0.02,
                                rtol=0.002,
                            ):
                                max_difference = (
                                    per_token_logps.detach()[valid_tokens]
                                    - old_logps[valid_tokens]
                                ).abs().max().item()
                                raise RuntimeError(
                                    "rollout and policy log-probabilities diverged before the "
                                    "optimizer step (maximum absolute difference "
                                    f"{max_difference:.6g}); check dropout or generation settings"
                                )

                            if self.beta > 0:
                                ref_logps = ref_per_token_logps[index]
                                delta = ref_logps - per_token_logps
                                per_token_kl = torch.exp(delta) - delta - 1.0
                            else:
                                per_token_kl = torch.zeros_like(per_token_logps)

                            ratio = torch.exp(per_token_logps - old_logps)
                            clipped_ratio = ratio.clamp(
                                1.0 - self.clip_epsilon,
                                1.0 + self.clip_epsilon,
                            )
                            token_count = candidate_mask.sum(dim=1).clamp_min(1.0)
                            advantage = advantages[index].view(1, 1)
                            surrogate = torch.minimum(
                                ratio * advantage,
                                clipped_ratio * advantage,
                            )
                            policy_loss = -(
                                surrogate * candidate_mask
                            ).sum(dim=1) / token_count
                            kl_loss = (
                                per_token_kl * candidate_mask
                            ).sum(dim=1) / token_count
                            candidate_loss = (policy_loss + self.beta * kl_loss).mean()

                        # Divide by group size here; training_step applies the
                        # prompt-batch and gradient-accumulation scaling.
                        backward_callback(candidate_loss / self.num_generations)

                    loss_total += candidate_loss.detach() / self.num_generations
                    kl_total += kl_loss.detach().mean() / self.num_generations
                    clipped_token_total += (
                        ((ratio.detach() - 1.0).abs() > self.clip_epsilon).float()
                        * candidate_mask
                    ).sum()
                    valid_token_total += candidate_mask.sum()
                    completion_length_total += token_count.detach().sum()

                    # Do not retain a current-policy graph or LM-head output while
                    # scoring the next completion.
                    del (
                        per_token_logps,
                        per_token_kl,
                        ratio,
                        clipped_ratio,
                        surrogate,
                        policy_loss,
                        kl_loss,
                        candidate_loss,
                        token_count,
                    )

            stats = {
                "reward": float(mean_reward.detach().item()),
                "reward_std": float(reward_std.detach().item()),
                "kl": float(kl_total.detach().item()),
                "clip_fraction": float(
                    (clipped_token_total / valid_token_total.clamp_min(1.0)).item()
                ),
                "completion_length": float(
                    (completion_length_total / self.num_generations).item()
                ),
            }
            for index, name in enumerate(self.reward_names):
                stats[f"rewards/{name}"] = float(
                    rewards_per_func[:, index].mean().detach().item()
                )
            return loss_total.detach(), stats
        finally:
            model.train(was_training)

    def _sample_loss_batched(
        self, model, sample: Mapping[str, object]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        was_training = model.training
        model.eval()
        try:
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

            # Save the behavior-policy log-probabilities before building the
            # current-policy graph, just as the rollout is generated.
            with torch.no_grad():
                old_per_token_logps = self._get_per_token_logps(
                    model,
                    full_ids,
                    attention_mask,
                    repeated_inputs,
                    prompt_length,
                )
            ref_per_token_logps = None
            if self.beta > 0:
                with torch.no_grad():
                    if self.ref_model is not None:
                        ref_per_token_logps = self._get_per_token_logps(
                            self.ref_model,
                            full_ids,
                            attention_mask,
                            repeated_inputs,
                            prompt_length,
                        )
                    elif self.reference_adapter_name is not None:
                        # Switch before constructing the policy graph. PEFT's
                        # set_adapter also toggles requires_grad on adapter weights.
                        base_model = self.accelerator.unwrap_model(model)
                        base_model.set_adapter(self.reference_adapter_name)
                        try:
                            ref_per_token_logps = self._get_per_token_logps(
                                model,
                                full_ids,
                                attention_mask,
                                repeated_inputs,
                                prompt_length,
                            )
                        finally:
                            base_model.set_adapter(self.policy_adapter_name)
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

            per_token_logps = self._get_per_token_logps(
                model,
                full_ids,
                attention_mask,
                repeated_inputs,
                prompt_length,
            )
            valid_tokens = completion_mask.bool()
            if valid_tokens.any() and not torch.allclose(
                per_token_logps.detach()[valid_tokens],
                old_per_token_logps[valid_tokens],
                atol=0.02,
                rtol=0.002,
            ):
                max_difference = (
                    per_token_logps.detach()[valid_tokens]
                    - old_per_token_logps[valid_tokens]
                ).abs().max().item()
                raise RuntimeError(
                    "rollout and policy log-probabilities diverged before the optimizer step "
                    f"(maximum absolute difference {max_difference:.6g}); check dropout or "
                    "generation settings"
                )

            if self.beta > 0:
                delta = ref_per_token_logps - per_token_logps
                per_token_kl = torch.exp(delta) - delta - 1.0
            else:
                per_token_kl = torch.zeros_like(per_token_logps)

            mean_reward = rewards.mean()
            reward_std = rewards.std(unbiased=False)
            advantages = (rewards - mean_reward) / (reward_std + 1e-4)
            ratio = torch.exp(per_token_logps - old_per_token_logps.detach())
            clipped_ratio = ratio.clamp(
                1.0 - self.clip_epsilon,
                1.0 + self.clip_epsilon,
            )
            advantages = advantages.detach().unsqueeze(1)
            surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)
            token_count = completion_mask.sum(dim=1).clamp_min(1.0)
            policy_loss = -(surrogate * completion_mask).sum(dim=1) / token_count
            kl_loss = (per_token_kl * completion_mask).sum(dim=1) / token_count
            loss = (policy_loss + self.beta * kl_loss).mean()
            clip_fraction = (
                ((ratio - 1.0).abs() > self.clip_epsilon).float() * completion_mask
            ).sum() / completion_mask.sum().clamp_min(1.0)

            stats = {
                "reward": float(mean_reward.detach().item()),
                "reward_std": float(reward_std.detach().item()),
                "kl": float(kl_loss.detach().mean().item()),
                "clip_fraction": float(clip_fraction.detach().item()),
                "completion_length": float(token_count.detach().mean().item()),
            }
            for index, name in enumerate(self.reward_names):
                stats[f"rewards/{name}"] = float(rewards_per_func[:, index].mean().detach().item())
            return loss, stats
        finally:
            model.train(was_training)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Stream completion graphs and accumulate their gradients immediately."""
        del num_items_in_batch
        model.train()
        optimizer = getattr(self, "optimizer", None)
        if hasattr(optimizer, "train") and callable(optimizer.train):
            optimizer.train()
        inputs = self._prepare_inputs(inputs)
        if isinstance(inputs, Mapping):
            samples = [inputs]
        elif isinstance(inputs, Sequence) and inputs:
            samples = list(inputs)
        else:
            raise ValueError("GRPO received an empty batch")

        gradient_accumulation_steps = int(
            getattr(
                self,
                "current_gradient_accumulation_steps",
                getattr(self.args, "gradient_accumulation_steps", 1),
            )
        )
        if gradient_accumulation_steps < 1:
            raise ValueError("gradient accumulation steps must be positive")

        batch_stats = defaultdict(list)
        detached_losses = []
        for sample_index, sample in enumerate(samples):
            is_last_sample = sample_index == len(samples) - 1

            def backward_callback(candidate_loss):
                self.accelerator.backward(
                    candidate_loss / (len(samples) * gradient_accumulation_steps)
                )

            def backward_context(is_last_candidate):
                if is_last_sample and is_last_candidate:
                    return nullcontext()
                no_sync = getattr(model, "no_sync", None)
                return no_sync() if callable(no_sync) else nullcontext()

            sample_loss, stats = self._sample_loss(
                model,
                sample,
                backward_callback=backward_callback,
                backward_context=backward_context,
            )
            detached_losses.append(sample_loss.detach())
            for key, value in stats.items():
                batch_stats[key].append(value)

        for key, values in batch_stats.items():
            self._metrics[key].append(sum(values) / len(values))
        return torch.stack(detached_losses).mean()

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
