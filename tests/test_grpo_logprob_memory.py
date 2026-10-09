import argparse
from types import MethodType, SimpleNamespace

import pytest
import torch
from accelerate import Accelerator
from accelerate.utils.operations import convert_outputs_to_fp32

from qwen3vl_sft.train.arguments import (
    DEFAULT_GRPO_COMPLETION_LENGTH,
    grpo_completion_length,
)
from qwen3vl_sft.train.grpo_trainer import MultimodalGRPOTrainer


class RecordingLogitsModel:
    def __init__(self, full_logits):
        self.full_logits = full_logits
        self.logits_to_keep = None
        self.returned_logits = None

    def __call__(self, **kwargs):
        self.logits_to_keep = kwargs["logits_to_keep"]
        self.returned_logits = self.full_logits[:, -self.logits_to_keep :, :]
        return SimpleNamespace(logits=self.returned_logits)


def test_logprob_forward_projects_only_completion_positions():
    trainer = object.__new__(MultimodalGRPOTrainer)
    trainer.max_completion_length = 512
    trainer._model_inputs = lambda inputs: dict(inputs)
    trainer._full_position_ids = lambda input_ids, attention_mask, model_inputs: None

    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [7, 8, 9, 10, 11, 12, 13]])
    attention_mask = torch.ones_like(input_ids)
    full_logits = torch.randn(2, 7, 17, dtype=torch.bfloat16)
    model = RecordingLogitsModel(full_logits)

    actual = trainer._get_per_token_logps(
        model,
        input_ids,
        attention_mask,
        {},
        prompt_length=3,
    )

    # Four completion tokens need their four predicting logits; the extra
    # retained position is the unused final sequence logit.
    assert model.logits_to_keep == 5
    assert model.returned_logits.shape == (2, 5, 17)
    reference_logits = full_logits[:, 2:-1, :].float()
    target_ids = input_ids[:, 3:]
    expected = torch.log_softmax(reference_logits, dim=-1).gather(
        -1, target_ids.unsqueeze(-1)
    ).squeeze(-1)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_completion_limit_defaults_to_512_and_accepts_configured_positive_values():
    assert DEFAULT_GRPO_COMPLETION_LENGTH == 512
    assert grpo_completion_length("512") == 512
    assert grpo_completion_length("513") == 513
    assert grpo_completion_length("1024") == 1024
    for value in ("0", "-1", "not-an-int"):
        with pytest.raises(argparse.ArgumentTypeError):
            grpo_completion_length(value)



def test_trainer_removes_accelerate_full_output_fp32_wrapper():
    class Bf16OutputModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones((), dtype=torch.bfloat16))

        def forward(self):
            return {"logits": self.weight.view(1)}

    model = Bf16OutputModel()
    model._original_forward = model.forward
    model.forward = MethodType(
        convert_outputs_to_fp32(model.forward.__func__),
        model,
    )
    assert model()["logits"].dtype == torch.float32

    trainer = object.__new__(MultimodalGRPOTrainer)
    trainer.accelerator = object.__new__(Accelerator)
    trainer._disable_accelerate_fp32_output_conversion(model)

    assert not hasattr(model, "_original_forward")
    assert model()["logits"].dtype == torch.bfloat16
