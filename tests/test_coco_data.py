import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from qwen3vl_sft.evaluation.coco_data import (
    CocoFrame,
    build_eval_records,
    build_fixed_support_eval_records,
    build_train_records,
    load_coco_frames,
)


def frames(prefix, count=5):
    return [
        CocoFrame(
            image_id=index,
            category_id=1,
            category="widget",
            image_path=f"/{prefix}-{index}.jpg",
            boxes=((100, 100, 400, 500),),
        )
        for index in range(count)
    ]


class CocoDataTest(unittest.TestCase):
    def test_train_builder_is_deterministic_and_balanced(self):
        values = {"widget": frames("train")}
        first = build_train_records(values, shots=[1, 2], num_samples=6, seed=7)
        second = build_train_records(values, shots=[1, 2], num_samples=6, seed=7)
        self.assertEqual(first, second)
        self.assertEqual({record["num_shots"] for record in first}, {1, 2})
        self.assertTrue(all(len(record["image"]) == record["num_shots"] + 1 for record in first))

    def test_eval_builder_keeps_query_images_unique(self):
        support = {"widget": frames("train")}
        query = {"widget": frames("val", count=4)}
        records = build_eval_records(support, query, shots=[1, 2], seed=4, num_query_images=4)
        self.assertEqual(len(records), 4)
        self.assertEqual(len({record["query"]["image_id"] for record in records}), 4)
        self.assertEqual({record["num_shots"] for record in records}, {1, 2})
        self.assertTrue(all(record["protocol"] == "positive_category_conditioned_icl" for record in records))

    def test_fixed_support_builder_keeps_short_support_categories(self):
        support = {
            "widget": frames("support", count=3),
            "gadget": frames("gadget", count=4),
        }
        query = {
            "widget": frames("widget-query", count=2),
            "gadget": frames("gadget-query", count=1),
        }
        records = build_fixed_support_eval_records(support, query, shot=4)
        self.assertEqual(len(records), 3)
        self.assertEqual({record["category"] for record in records}, {"widget", "gadget"})
        self.assertEqual({len(record["support"]) for record in records}, {3, 4})
        self.assertEqual(len({record["id"] for record in records}), len(records))

    def test_coco_json_loader_and_cli_builder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_root = root / "images"
            image_root.mkdir()
            for name in ("one.jpg", "two.jpg"):
                (image_root / name).touch()
            annotations = {
                "images": [
                    {"id": 1, "file_name": "one.jpg", "width": 100, "height": 100},
                    {"id": 2, "file_name": "two.jpg", "width": 100, "height": 100},
                ],
                "categories": [{"id": 7, "name": "widget"}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 7, "bbox": [10, 20, 30, 40]},
                    {"id": 2, "image_id": 2, "category_id": 7, "bbox": [20, 10, 20, 30]},
                ],
            }
            annotation_path = root / "instances.json"
            annotation_path.write_text(json.dumps(annotations), encoding="utf-8")

            loaded = load_coco_frames(annotation_path, image_root, min_box_area_ratio=0.01)
            self.assertEqual([frame.image_id for frame in loaded["widget"]], [1, 2])
            self.assertEqual(loaded["widget"][0].boxes, ((100, 200, 400, 600),))
            self.assertEqual((loaded["widget"][0].width, loaded["widget"][0].height), (100, 100))

            output_path = root / "train.json"
            cli = Path(__file__).parents[1] / "tools" / "build_coco.py"
            subprocess.run(
                [
                    sys.executable,
                    str(cli),
                    "train",
                    "--annotations",
                    str(annotation_path),
                    "--image-root",
                    str(image_root),
                    "--output",
                    str(output_path),
                    "--shots",
                    "1",
                    "--num-samples",
                    "1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["records"]), 1)
            self.assertEqual(payload["records"][0]["loss_mode"], "last_assistant")


if __name__ == "__main__":
    unittest.main()
