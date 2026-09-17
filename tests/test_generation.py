import json
import tempfile
import unittest
from pathlib import Path

from qwen3vl_sft.evaluation.coco_protocol import PROTOCOL_NAME, PROMPT_TEMPLATE_VERSION
from qwen3vl_sft.evaluation.generation import load_episodes
from qwen3vl_sft.evaluation.metrics import evaluate_episode_predictions


class GenerationTest(unittest.TestCase):
    def test_episode_paths_are_resolved_relative_to_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "episodes.json"
            manifest.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "id": "episode",
                                "protocol": PROTOCOL_NAME,
                                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                                "category": "widget",
                                "num_shots": 0,
                                "support": [],
                                "query": {"image": "images/query.jpg", "boxes": []},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            _, records = load_episodes(manifest)
            self.assertEqual(records[0]["query"]["image"], str(root / "images/query.jpg"))

    def test_metric_input_lengths_must_match(self):
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            evaluate_episode_predictions(
                [
                    {
                        "category": "widget",
                        "num_shots": 0,
                        "query": {"boxes": []},
                    }
                ],
                [],
            )

    def test_metric_input_lengths_match_for_empty_result(self):
        result = evaluate_episode_predictions(
            [
                {
                    "category": "widget",
                    "num_shots": 0,
                    "query": {"boxes": []},
                }
            ],
            ["[]"],
        )
        self.assertEqual(result["metrics_by_shot"]["0"]["count_accuracy"], 1.0)

    def test_map_uses_official_coco_annotations(self):
        try:
            import pycocotools  # noqa: F401
        except ImportError:
            self.skipTest("pycocotools is required for official COCO mAP")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotations = root / "instances.json"
            annotations.write_text(
                json.dumps(
                    {
                        "images": [{"id": 17, "file_name": "image.jpg", "width": 100, "height": 100}],
                        "categories": [{"id": 7, "name": "widget"}],
                        "annotations": [
                            {
                                "id": 3,
                                "image_id": 17,
                                "category_id": 7,
                                "bbox": [10, 20, 30, 40],
                                "area": 1200,
                                "iscrowd": 0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            result = evaluate_episode_predictions(
                [
                    {
                        "category": "widget",
                        "num_shots": 1,
                        # Deliberately inconsistent with the official GT: mAP
                        # must use instances.json, not this convenience field.
                        "query": {
                            "image_id": 17,
                            "category_id": 7,
                            "width": 100,
                            "height": 100,
                            "boxes": [[0, 0, 100, 100]],
                        },
                    }
                ],
                ['[{"bbox_2d":[100,200,400,600],"label":"widget"}]'],
                coco_annotations_path=str(annotations),
            )
            metrics = result["metrics_by_shot"]["1"]["coco_map"]["model"]
            self.assertAlmostEqual(metrics["map_50_95"], 1.0)
            self.assertAlmostEqual(metrics["map_50"], 1.0)


if __name__ == "__main__":
    unittest.main()
