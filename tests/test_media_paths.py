import json
import tempfile
import unittest
from pathlib import Path

from qwen3vl_sft.data.messages import build_messages
from qwen3vl_sft.data.paths import resolve_media_path
from qwen3vl_sft.evaluation.coco.generation import load_episodes


class MediaPathResolutionTest(unittest.TestCase):
    def test_existing_absolute_path_is_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "already-local.jpg"
            image.write_bytes(b"placeholder")

            self.assertEqual(resolve_media_path(str(image), root), str(image.resolve()))

    def test_training_messages_relocate_stale_coco_absolute_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            image = data_root / "COCO" / "train2017" / "000000222548.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"placeholder")

            stale_path = (
                "/home/u1120240334/data/LLM/.modelscope-cache/modelscope/"
                "COCO2017_Instance_Segmentation/master/data_files/extracted/"
                "319c485e918ea37566b91c242c088071a17d3f81651bc18ed1fa4e6cc985dec0/"
                "COCO2017train/train2017/000000222548.jpg"
            )
            record = {
                "image": [stale_path],
                "conversations": [
                    {"from": "human", "value": "<image>"},
                    {"from": "gpt", "value": "[]"},
                ],
            }

            messages = build_messages(record, base_path=data_root)

            self.assertEqual(messages[0]["content"][0]["image"], str(image.resolve()))

    def test_eval_manifest_relocates_stale_images_and_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            support_image = data_root / "COCO" / "train2017" / "000000000001.jpg"
            query_image = data_root / "COCO" / "val2017" / "000000000002.jpg"
            support_annotations = (
                data_root / "COCO" / "annotations" / "instances_train2017.json"
            )
            annotations = data_root / "COCO" / "annotations" / "instances_val2017.json"
            for path in (support_image, query_image, support_annotations, annotations):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"placeholder")

            manifest = root / "episodes.json"
            manifest.write_text(
                json.dumps(
                    {
                        "support_annotations": (
                            "/home/u1120240334/data/LLM/.modelscope-cache/modelscope/"
                            "COCO2017_Instance_Segmentation/master/data_files/extracted/"
                            "319c485e918ea37566b91c242c088071a17d3f81651bc18ed1fa4e6cc985dec0/"
                            "COCO2017train/annotations/instances_train2017.json"
                        ),
                        "query_annotations": (
                            "/home/u1120240334/data/LLM/.modelscope-cache/modelscope/"
                            "COCO2017_Instance_Segmentation/master/data_files/extracted/"
                            "7b9623c81b9b032bc53f694abf10370f86bdbb6ac77b0eef46b65816741e32f4/"
                            "COCO2017val/annotations/instances_val2017.json"
                        ),
                        "records": [
                            {
                                "protocol": "positive_category_conditioned_icl",
                                "prompt_template_version": "inst-v5",
                                "support": [
                                    {
                                        "image": (
                                            "/home/u1120240334/data/LLM/.modelscope-cache/"
                                            "modelscope/COCO2017_Instance_Segmentation/"
                                            "master/data_files/extracted/319c485e918ea37566b91c242c088071a17d3f81651bc18ed1fa4e6cc985dec0/"
                                            "COCO2017train/train2017/000000000001.jpg"
                                        )
                                    }
                                ],
                                "query": {
                                    "image": (
                                        "/home/u1120240334/data/LLM/.modelscope-cache/"
                                        "modelscope/COCO2017_Instance_Segmentation/"
                                        "master/data_files/extracted/7b9623c81b9b032bc53f694abf10370f86bdbb6ac77b0eef46b65816741e32f4/"
                                        "COCO2017val/val2017/000000000002.jpg"
                                    )
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            metadata, records = load_episodes(manifest, media_root=data_root)

            self.assertEqual(metadata["query_annotations"], str(annotations.resolve()))
            self.assertEqual(
                metadata["support_annotations"], str(support_annotations.resolve())
            )
            self.assertEqual(records[0]["support"][0]["image"], str(support_image.resolve()))
            self.assertEqual(records[0]["query"]["image"], str(query_image.resolve()))

    def test_missing_media_reports_data_root_guidance(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "DATA_ROOT"):
                resolve_media_path(
                    "/home/other/COCO2017train/train2017/missing.jpg",
                    Path(directory) / "data",
                )


if __name__ == "__main__":
    unittest.main()
