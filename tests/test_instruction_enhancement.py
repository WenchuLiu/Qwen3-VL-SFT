import json
import tempfile
import unittest
from pathlib import Path

from qwen3vl_sft.evaluation.coco.protocol import (
    build_eval_messages,
    build_question,
    load_category_descriptions,
)


class InstructionEnhancementTest(unittest.TestCase):
    def setUp(self):
        self.record = {
            "category": "fish",
            "support": [
                {"image": "/tmp/support.jpg", "boxes": [[10, 20, 300, 400]]},
            ],
            "query": {"image": "/tmp/query.jpg", "boxes": []},
        }

    def test_baseline_question_is_unchanged_without_description(self):
        self.assertEqual(
            build_question("fish"),
            'Locate all of the following objects: fish in the image and output all '
            'detections as a JSON list like [{"bbox_2d":[x1,y1,x2,y2],"label":"class_name"}].',
        )

    def test_zero_shot_baseline_is_a_normal_detection_prompt(self):
        record = {**self.record, "support": []}
        messages = build_eval_messages(
            record,
            min_pixels=1,
            max_pixels=100,
        )
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"][0]["type"], "text")
        prompt = messages[0]["content"][0]["text"]
        self.assertNotIn("in-context", prompt.lower())
        self.assertNotIn("preceding", prompt.lower())
        self.assertIn('Identify and localize all instances of "fish" in the image.', prompt)
        self.assertIn('"label": "fish"', prompt)
        self.assertIn('"score": 0.95', prompt)
        self.assertIn("0.0 to 1.0", prompt)

    def test_zero_shot_ie_has_detpo_instructions_without_context_wording(self):
        record = {**self.record, "support": []}
        messages = build_eval_messages(
            record,
            min_pixels=1,
            max_pixels=100,
            instruction_enhancement=True,
            category_descriptions={"fish": "an aquatic animal with fins and scales"},
        )
        self.assertEqual(len(messages), 1)
        prompt = messages[0]["content"][0]["text"]
        self.assertNotIn("in-context", prompt.lower())
        self.assertNotIn("preceding", prompt.lower())
        self.assertIn("Follow these annotator instructions", prompt)
        self.assertIn("an aquatic animal with fins and scales", prompt)
        self.assertIn('"score": 0.95', prompt)

    def test_description_uses_detpo_block_only_on_query(self):
        messages = build_eval_messages(
            self.record,
            min_pixels=1,
            max_pixels=100,
            instruction_enhancement=True,
            category_descriptions={"fish": "an aquatic animal with fins and scales"},
        )
        texts = [
            item["text"]
            for message in messages
            for item in message["content"]
            if item.get("type") == "text"
        ]
        self.assertEqual(len(texts), 4)
        _, support_text, answer_text, query_text = texts
        self.assertNotIn("annotator instructions", support_text)
        self.assertNotIn("an aquatic animal with fins and scales", support_text)
        self.assertEqual(answer_text, '[{"bbox_2d":[10,20,300,400],"label":"fish"}]')
        self.assertIn('Identify and localize all instances of "fish" in the query image.', query_text)
        self.assertIn("Follow these annotator instructions to improve detection accuracy:", query_text)
        self.assertIn("an aquatic animal with fins and scales", query_text)
        self.assertIn('"label": "fish"', query_text)
        self.assertNotIn("visual description:", query_text)

    def test_episode_description_is_used_without_external_file(self):
        record = {**self.record, "category_description": "a small aquatic animal"}
        messages = build_eval_messages(
            record,
            min_pixels=1,
            max_pixels=100,
            instruction_enhancement=True,
        )
        self.assertIn("a small aquatic animal", messages[-1]["content"][1]["text"])

    def test_description_file_accepts_wrapped_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "descriptions.json"
            path.write_text(
                json.dumps({"descriptions": {"fish": "an aquatic animal"}}),
                encoding="utf-8",
            )
            self.assertEqual(
                load_category_descriptions(path),
                {"fish": "an aquatic animal"},
            )


if __name__ == "__main__":
    unittest.main()
