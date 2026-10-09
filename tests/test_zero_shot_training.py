import json

from PIL import Image

from qwen3vl_sft.evaluation.coco.protocol import (
    ZERO_SHOT_EVAL_SYSTEM_PROMPT, ZERO_SHOT_TRAIN_SYSTEM_PROMPT,
    build_question, build_sft_record,
)
from qwen3vl_sft.train.grpo_data import make_grpo_sample
from tools.prepare_coco_zero_shot import convert_record


def test_zero_shot_sft_and_grpo_have_no_support_or_target_leakage(tmp_path):
    image = tmp_path / '000000000001.jpg'
    Image.new('RGB', (32, 32)).save(image)
    frame = {'image': str(image), 'boxes': [[123, 234, 567, 789]]}
    record = build_sft_record(record_id='zero', category='cat', support=[], query=frame)
    assert record['num_shots'] == 0
    assert len(record['image']) == 1
    assert len(record['conversations']) == 3
    assert record['conversations'][0]['value'] == ZERO_SHOT_TRAIN_SYSTEM_PROMPT
    assert 'preceding' not in record['conversations'][1]['value']
    sample = make_grpo_sample(record, base_path=tmp_path)
    assert len(sample['prompt']) == 2
    assert sample['prompt'][0]['content'][0]['text'] == ZERO_SHOT_EVAL_SYSTEM_PROMPT
    text = sample['prompt'][-1]['content'][-1]['text']
    assert text == build_question('cat', include_confidence=True)
    assert 'preceding' not in text
    assert 'score' in text
    assert '123' not in json.dumps(sample['prompt'])
    assert sample['target_boxes'] == frame['boxes']


def test_zero_shot_builder_needs_only_one_query_image():
    from qwen3vl_sft.evaluation.coco.data import CocoFrame, build_train_records, parse_shots

    frame = CocoFrame(image_id=1, category_id=1, category='cat',
                      image_path='/unused/1.jpg', width=32, height=32,
                      boxes=((123, 234, 567, 789),))
    records = build_train_records({'cat': [frame]}, shots=parse_shots(['0'], allow_zero=True),
                                  num_samples=1, seed=42)
    assert len(records) == 1
    assert records[0]['num_shots'] == 0
    assert records[0]['image'] == ['/unused/1.jpg']


def test_conversion_preserves_queries_and_answers_and_rejects_other_split(tmp_path):
    import pytest

    support = tmp_path / '000000000002.jpg'
    query = tmp_path / '000000000001.jpg'
    Image.new('RGB', (32, 32)).save(query)
    record = build_sft_record(
        record_id='source', category='cat',
        support=[{'image': str(support), 'boxes': [[1, 2, 3, 4]]}],
        query={'image': str(query), 'boxes': [[123, 234, 567, 789]]},
    )
    converted = convert_record(record, data_root=tmp_path, selected_ids={1})
    assert converted['id'] == record['id']
    assert converted['image'] == [str(query)]
    assert converted['conversations'][-1] == record['conversations'][-1]
    assert converted['num_shots'] == 0
    with pytest.raises(ValueError, match='outside'):
        convert_record(record, data_root=tmp_path, selected_ids={2})


def test_segment_boundary_saves_without_changing_scheduler_horizon():
    from types import SimpleNamespace
    from transformers.trainer_callback import TrainerControl
    from qwen3vl_sft.train.grpo_runner import StopAfterStepCallback

    args = SimpleNamespace(max_steps=1000)
    callback = StopAfterStepCallback(250)
    control = TrainerControl()
    callback.on_step_end(args, SimpleNamespace(global_step=249), control)
    assert not control.should_training_stop
    callback.on_step_end(args, SimpleNamespace(global_step=250), control)
    assert control.should_training_stop
    assert control.should_save
    assert args.max_steps == 1000


def test_grpo_resume_restores_root_policy_as_well_as_reference(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from transformers import Trainer
    from qwen3vl_sft.train.grpo_trainer import MultimodalGRPOTrainer

    (tmp_path / 'adapter_config.json').write_text('{}')
    reference = tmp_path / 'grpo_reference'
    reference.mkdir()
    (reference / 'adapter_config.json').write_text('{}')
    parent_load = Mock()
    monkeypatch.setattr(Trainer, '_load_from_checkpoint', parent_load)
    model = Mock()
    trainer = object.__new__(MultimodalGRPOTrainer)
    trainer.model = model
    trainer.policy_adapter_name = 'default'
    trainer.reference_adapter_name = 'grpo_reference'
    trainer._load_from_checkpoint(str(tmp_path))
    parent_load.assert_called_once_with(str(tmp_path), model=None)
    model.load_adapter.assert_called_once_with(str(tmp_path), 'default', is_trainable=True)
    model.set_adapter.assert_called_once_with('default')
