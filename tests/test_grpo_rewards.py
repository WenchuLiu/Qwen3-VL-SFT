import pytest

from qwen3vl_sft.train.rewards import REWARD_FUNCS_REGISTRY, box_iou_reward


def _completion(entries):
    import json

    return json.dumps(entries)


def test_box_iou_reward_uses_best_iou_precision_recall_f1():
    completion = _completion([
        {"bbox_2d": [0, 0, 10, 10], "label": "cat"},
        {"bbox_2d": [20, 20, 30, 30], "label": "cat"},
    ])
    reward = box_iou_reward(
        [completion],
        target_boxes=[[0, 0, 10, 10]],
        target_labels=[["cat"]],
        category=["cat"],
    )
    # Precision=(1+0)/2, recall=1, so the harmonic mean is 2/3.
    assert reward == pytest.approx([2.0 / 3.0])


def test_box_iou_reward_matches_each_box_independently_without_cutoff():
    completion = _completion([
        {"bbox_2d": [0, 0, 10, 10], "label": "cat"},
        {"bbox_2d": [0, 0, 10, 10], "label": "cat"},
    ])
    reward = box_iou_reward(
        [completion],
        target_boxes=[[0, 0, 10, 10]],
        target_labels=[["cat"]],
        category=["cat"],
        iou_threshold=1.0,
    )
    # Rex-Omni computes a best IoU independently for each prediction and GT.
    assert reward == pytest.approx([1.0])


def test_box_iou_reward_requires_class_compatibility():
    completion = _completion([
        {"bbox_2d": [0, 0, 10, 10], "label": "dog"},
    ])
    reward = box_iou_reward(
        [completion],
        target_boxes=[[0, 0, 10, 10]],
        target_labels=[["cat"]],
        category=["cat"],
    )
    assert reward == [0.0]


def test_box_iou_reward_empty_cases_match_rex_omni():
    assert box_iou_reward(["[]"], target_boxes=[[]]) == [1.0]
    false_positive = _completion([
        {"bbox_2d": [0, 0, 10, 10], "label": "cat"},
    ])
    assert box_iou_reward([false_positive], target_boxes=[[]]) == [0.0]


def test_box_iou_reward_accepts_reversed_coordinates_like_rex_omni():
    completion = _completion([
        {"bbox_2d": [10, 10, 0, 0], "label": "cat"},
    ])
    reward = box_iou_reward(
        [completion],
        target_boxes=[[0, 0, 10, 10]],
        target_labels=[["cat"]],
        category=["cat"],
    )
    assert reward == [1.0]
    assert REWARD_FUNCS_REGISTRY["box_iou"] is box_iou_reward


@pytest.mark.parametrize('completion', [
    "[{{'bbox_2d': [0, 0, 10, 10], 'label': 'cat', 'score': 0.9}}]",
    '[{[0, 0, 10, 10]}]',
    "{{'label': 'cat'}: 1}",
])
def test_malformed_unhashable_literals_do_not_crash_rollout_rewards(completion):
    from qwen3vl_sft.train.rewards import format_reward, parse_target_boxes, score_reward

    kwargs = dict(target_boxes=[[0, 0, 10, 10]], target_labels=[['cat']], category=['cat'])
    assert box_iou_reward([completion], **kwargs) == [0.0]
    assert score_reward([completion], **kwargs) == [0.0]
    assert format_reward([completion]) == [0.0]
    assert parse_target_boxes(completion) == []
