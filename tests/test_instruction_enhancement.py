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

    def test_zero_shot_uses_standalone_eval_prompt(self):
        record = {**self.record, "support": []}
        messages = build_eval_messages(record, min_pixels=1, max_pixels=100)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertNotIn("in-context examples", messages[0]["content"][0]["text"])
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"][0]["type"], "image")
        self.assertEqual(messages[1]["content"][1]["type"], "text")
        self.assertNotIn(
            "Using the preceding in-context examples",
            messages[1]["content"][1]["text"],
        )
        self.assertIn("Locate all of the following objects: fish in the image", messages[1]["content"][1]["text"])
        self.assertIn('"score":0.95', messages[1]["content"][1]["text"])

    def test_zero_shot_ie_uses_separate_category_definition_sentence(self):
        record = {**self.record, "support": []}
        messages = build_eval_messages(
            record,
            min_pixels=1,
            max_pixels=100,
            instruction_enhancement=True,
            category_descriptions={
                "fish": "an aquatic animal with fins and scales",
            },
        )
        query_text = messages[-1]["content"][1]["text"]
        self.assertIn(
            "Locate all of the following objects: fish in the image.\n"
            "fish is an aquatic animal with fins and scales.",
            query_text,
        )
        self.assertNotIn("fish —", query_text)
        self.assertNotIn("Using the preceding in-context examples", query_text)

    def test_few_shot_retains_v5_eval_prompt(self):
        messages = build_eval_messages(self.record, min_pixels=1, max_pixels=100)
        self.assertIn("in-context examples", messages[0]["content"][0]["text"])
        self.assertIn(
            "Using the preceding in-context examples",
            messages[-1]["content"][1]["text"],
        )

    def test_description_is_added_only_to_final_query_question(self):
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
        described_texts = [
            text
            for text in texts
            if "fish is an aquatic animal with fins and scales" in text
        ]
        self.assertEqual(len(described_texts), 1)
        self.assertIn("fish in the image and output", texts[1])
        self.assertNotIn("fish is an aquatic animal", texts[1])
        self.assertIn("fish in the query image.", texts[-1])

    def test_detpo_uses_original_single_image_prompt(self):
        record = {
            **self.record,
            "category_description": "a small aquatic animal with a streamlined body",
        }
        messages = build_eval_messages(
            record,
            min_pixels=1,
            max_pixels=100,
            detpo=True,
        )
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"][0]["type"], "text")
        self.assertEqual(messages[0]["content"][1]["type"], "image")
        self.assertEqual(messages[0]["content"][1]["image"], "/tmp/query.jpg")
        query_text = messages[0]["content"][0]["text"]
        self.assertIn(
            'Identify and localize all instances of "fish" in the image.',
            query_text,
        )
        self.assertIn("Output Requirements:", query_text)
        self.assertIn("Include at most 20 detections.", query_text)
        self.assertIn(
            "Follow these annotator instructions to improve detection accuracy:",
            query_text,
        )
        self.assertIn(
            "a small aquatic animal with a streamlined body",
            query_text,
        )
        self.assertIn('"score": 0.95', query_text)
        self.assertIn("0-1000 normalized coordinates", query_text)
        self.assertNotIn("fish is a small aquatic animal", query_text)
        self.assertNotIn("in-context examples", query_text)

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
