import json
import unittest

from qwen3vl_sft.evaluation.coco_protocol import (
    EVAL_SYSTEM_PROMPT,
    LOSS_MODE,
    PROMPT_TEMPLATE_VERSION,
    TRAIN_SYSTEM_PROMPT,
    build_eval_messages,
    build_sft_record,
)
from qwen3vl_sft.evaluation.metrics import (
    evaluate_episode_predictions,
    parse_detection_output,
    parse_scored_detection_output,
)


def episode():
    return {
        "id": "episode-1",
        "protocol": "positive_category_conditioned_icl",
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "category": "widget",
        "num_shots": 1,
        "support": [{"image_id": 1, "image": "/tmp/support.jpg", "boxes": [[100, 100, 400, 500]]}],
        "query": {"image_id": 2, "image": "/tmp/query.jpg", "boxes": [[200, 200, 600, 700]]},
    }


class ProtocolTest(unittest.TestCase):
    def test_inst_v5_is_the_default_prompt_protocol(self):
        self.assertEqual(PROMPT_TEMPLATE_VERSION, "inst-v5")

    def test_sft_omits_confidence_and_eval_requests_it(self):
        value = episode()
        record = build_sft_record(
            record_id=value["id"],
            category=value["category"],
            support=value["support"],
            query=value["query"],
        )
        self.assertEqual(record["loss_mode"], LOSS_MODE)
        self.assertEqual(record["num_shots"], 1)
        self.assertEqual(record["image"], ["/tmp/support.jpg", "/tmp/query.jpg"])
        self.assertEqual(
            record["conversations"][0],
            {"from": "system", "value": TRAIN_SYSTEM_PROMPT},
        )
        self.assertTrue(record["conversations"][1]["value"].startswith("<image>\nLocate"))
        self.assertTrue(record["conversations"][3]["value"].startswith("<image>\nUsing the preceding"))
        self.assertNotIn("score", json.dumps(record["conversations"]))
        self.assertNotIn("20 detections", json.dumps(record["conversations"]))

        messages = build_eval_messages(value, min_pixels=3136, max_pixels=640000)
        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[0]["content"][0]["text"], EVAL_SYSTEM_PROMPT)
        self.assertEqual(messages[1]["content"][1]["text"], record["conversations"][1]["value"][len("<image>\n"):])
        self.assertNotIn("score", messages[2]["content"][0]["text"])
        self.assertIn('"score":0.95', messages[3]["content"][1]["text"])
        self.assertIn("descending confidence", messages[3]["content"][1]["text"])
        self.assertNotIn("600", json.dumps(messages[-1]))
        self.assertNotIn("20 detections", json.dumps(messages))

    def test_zero_shot_generation_keeps_the_same_query_protocol(self):
        value = episode()
        value["support"] = []
        messages = build_eval_messages(value, min_pixels=3136, max_pixels=640000)
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("Using the preceding in-context examples", messages[-1]["content"][1]["text"])

    def test_parser_accepts_empty_json_and_fenced_grounding_json(self):
        self.assertEqual(parse_detection_output("[]"), [])
        text = "```json\n[{\"bbox_2d\":[1,2,300,400],\"label\":\"widget\"}]\n```"
        self.assertEqual(parse_detection_output(text), [("widget", [1.0, 2.0, 300.0, 400.0])])
        self.assertEqual(parse_detection_output("(widget, 1, 2, 300, 400);"), [("widget", [1.0, 2.0, 300.0, 400.0])])

    def test_parser_uses_detpo_score_and_missing_score_fallback(self):
        scored = json.dumps(
            [
                {"bbox_2d": [1, 2, 30, 40], "label": "widget", "score": 0.8},
                {"bbox_2d": [5, 6, 50, 60], "label": "widget"},
            ]
        )
        self.assertEqual(
            parse_scored_detection_output(scored),
            [
                ("widget", [1.0, 2.0, 30.0, 40.0], 0.8),
                ("widget", [5.0, 6.0, 50.0, 60.0], 0.5),
            ],
        )

    def test_metrics_are_perfect_for_the_exact_response(self):
        records = [episode()]
        response = json.dumps([{"bbox_2d": [200, 200, 600, 700], "label": "widget"}])
        result = evaluate_episode_predictions(records, [response])
        summary = result["metrics_by_shot"]["1"]
        self.assertEqual(summary["episodes"], 1)
        self.assertEqual(summary["count_accuracy"], 1.0)
        self.assertEqual(summary["f1_mean_over_iou"], 1.0)
        self.assertAlmostEqual(summary["coco_map"]["model"]["map_50_95"], 1.0)

    def test_coco_map_uses_model_score_and_detection_ranking(self):
        records = [episode()]
        response = json.dumps(
            [
                {"bbox_2d": [0, 0, 100, 100], "label": "widget", "score": 0.1},
                {"bbox_2d": [200, 200, 600, 700], "label": "widget", "score": 0.9},
            ]
        )
        summary = evaluate_episode_predictions(records, [response])["metrics_by_shot"]["1"]
        self.assertAlmostEqual(summary["coco_map"]["model"]["map_50_95"], 1.0)
        self.assertAlmostEqual(summary["coco_map"]["ranking"]["map_50_95"], 0.5)


if __name__ == "__main__":
    unittest.main()
