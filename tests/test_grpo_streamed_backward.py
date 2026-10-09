from collections import defaultdict
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from qwen3vl_sft.train.grpo_trainer import MultimodalGRPOTrainer


class CheckpointProbe(nn.Module):
    gradient_checkpointing = True

    def __init__(self):
        super().__init__()
        self._gradient_checkpointing_func = lambda function, *args: function(*args)


class ToyPolicy(nn.Module):
    def __init__(self, bias, context_state=None):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))
        self.context_state = context_state
        self.checkpoint_probe = CheckpointProbe()
        self.checkpoint_states = []
        self.events = []
        self.no_sync_calls = 0

    def forward(self, input_ids, logits_to_keep, **kwargs):
        del kwargs
        if torch.is_grad_enabled():
            self.checkpoint_states.append((self.training, self.checkpoint_probe.training))
            self.events.append(
                (
                    "forward",
                    bool(
                        self.context_state
                        and (
                            self.context_state["active"]
                            or self.context_state.get("no_sync", False)
                        )
                    ),
                )
            )
        logits = self.bias.view(1, 1, -1).expand(
            input_ids.shape[0], logits_to_keep, -1
        )
        return SimpleNamespace(logits=logits)

    @contextmanager
    def no_sync(self):
        self.no_sync_calls += 1
        if self.context_state is not None:
            self.context_state["no_sync"] = True
        try:
            yield
        finally:
            if self.context_state is not None:
                self.context_state["no_sync"] = False


class ToyProcessor:
    def __init__(self):
        self.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=4)

    def batch_decode(self, token_ids, **kwargs):
        del kwargs
        return [str(row.tolist()) for row in token_ids]


def make_trainer(context_state=None):
    trainer = object.__new__(MultimodalGRPOTrainer)
    trainer.processor = ToyProcessor()
    trainer.num_generations = 3
    trainer.max_completion_length = 4
    trainer.clip_epsilon = 0.2
    trainer.beta = 0.03
    trainer.reward_names = ["reward"]
    trainer.ref_model = ToyPolicy([0.3, -0.1, 0.2, 0.0, 0.4])
    trainer.reference_adapter_name = None
    trainer.policy_adapter_name = None
    trainer._metrics = defaultdict(list)
    trainer.compute_loss_context_manager = lambda: nullcontext()
    trainer._model_inputs = lambda inputs: dict(inputs)
    trainer._full_position_ids = (
        lambda input_ids, attention_mask, model_inputs: torch.zeros_like(input_ids)
    )
    trainer._render_prompt = lambda sample: {
        "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
    }
    sequences = torch.tensor(
        [
            [1, 2, 2, 4, 0],
            [1, 2, 3, 1, 4],
            [1, 2, 3, 4, 0],
        ],
        dtype=torch.long,
    )
    trainer._generate_completions = lambda model, prompt_inputs: sequences
    trainer._compute_rewards = lambda sample, prompts, completions: (
        torch.tensor([[0.3], [-0.2], [0.7]], dtype=torch.float32),
        torch.tensor([0.3, -0.2, 0.7], dtype=torch.float32),
    )
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda model: model)
    return trainer


def make_policy(context_state=None):
    return ToyPolicy([0.1, -0.3, 0.2, 0.5, -0.2], context_state=context_state)


def test_streamed_completion_backward_matches_group_loss_and_gradients():
    sample = {"prompt": [{"role": "user", "content": "locate"}]}

    batched_trainer = make_trainer()
    batched_policy = make_policy()
    batched_loss, batched_stats = batched_trainer._sample_loss_batched(
        batched_policy, sample
    )
    batched_loss.backward()
    expected_grad = batched_policy.bias.grad.detach().clone()

    context_state = {"active": False}
    streamed_trainer = make_trainer(context_state)
    streamed_policy = make_policy(context_state)
    context_flags = []

    @contextmanager
    def candidate_context(is_last):
        context_flags.append(is_last)
        assert not context_state["active"]
        context_state["active"] = True
        try:
            yield
        finally:
            context_state["active"] = False

    def backward(loss):
        assert context_state["active"]
        streamed_policy.events.append(("backward", True))
        loss.backward()

    streamed_loss, streamed_stats = streamed_trainer._sample_loss(
        streamed_policy,
        sample,
        backward_callback=backward,
        backward_context=candidate_context,
    )

    assert torch.allclose(streamed_loss, batched_loss.detach(), atol=1e-6, rtol=1e-6)
    assert torch.allclose(streamed_policy.bias.grad, expected_grad, atol=1e-6, rtol=1e-6)
    for key, value in batched_stats.items():
        assert streamed_stats[key] == pytest.approx(value, abs=1e-6)
    assert context_flags == [False, False, True]
    assert streamed_policy.checkpoint_states == [(False, True)] * 3
    assert streamed_policy.events == [
        ("forward", True),
        ("backward", True),
        ("forward", True),
        ("backward", True),
        ("forward", True),
        ("backward", True),
    ]


def test_training_step_scales_accumulation_and_syncs_only_final_completion():
    sample = {"prompt": [{"role": "user", "content": "locate"}]}
    context_state = {"active": False, "no_sync": False}
    trainer = make_trainer(context_state)
    policy = make_policy(context_state)
    trainer.optimizer = None
    trainer.args = SimpleNamespace(gradient_accumulation_steps=2)
    trainer.current_gradient_accumulation_steps = 2
    trainer._prepare_inputs = lambda inputs: inputs
    def backward(loss):
        policy.events.append(("backward", context_state["no_sync"]))
        loss.backward()

    trainer.accelerator.backward = backward

    reference_trainer = make_trainer()
    reference_policy = make_policy()
    reference_loss, _ = reference_trainer._sample_loss_batched(
        reference_policy, sample
    )
    reference_loss.backward()

    loss = trainer.training_step(policy, [sample])

    assert torch.allclose(loss, reference_loss.detach(), atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        policy.bias.grad,
        reference_policy.bias.grad / 2,
        atol=1e-6,
        rtol=1e-6,
    )
    assert policy.no_sync_calls == 2
    assert policy.events == [
        ("forward", True),
        ("backward", True),
        ("forward", True),
        ("backward", True),
        ("forward", False),
        ("backward", False),
    ]
