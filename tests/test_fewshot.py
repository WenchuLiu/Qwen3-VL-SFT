import json
import tempfile
import unittest
from pathlib import Path

from tools.evaluate_fewshot import (
    DATASET_MAX_NEW_TOKENS,
    DEFAULT_DATASETS,
    _max_new_tokens_for_dataset,
    _result_is_complete,
)


class FewshotConfigTest(unittest.TestCase):
    def test_matches_qwen3_vl_dataset_generation_budgets(self):
        self.assertEqual(
            DEFAULT_DATASETS,
            ("ArTaxOr", "Clipart1k", "FISH", "NEU-DET", "UODD", "VISUALDIOR"),
        )
        self.assertEqual(
            [DATASET_MAX_NEW_TOKENS[name] for name in DEFAULT_DATASETS],
            [1024, 1024, 1024, 1024, 1024, 2048],
        )
        self.assertEqual(_max_new_tokens_for_dataset("DIOR", None), 2048)
        self.assertEqual(_max_new_tokens_for_dataset("VISUALDIOR", None), 2048)
        self.assertEqual(_max_new_tokens_for_dataset("FISH", 2048), 2048)

    def test_skip_existing_rejects_result_with_old_generation_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            path.write_text(
                json.dumps(
                    {
                        "metrics_by_shot": {"1": {}},
                        "predictions": [],
                        "max_new_tokens": 256,
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(_result_is_complete(path, 1024))
            self.assertTrue(_result_is_complete(path, 256))


if __name__ == "__main__":
    unittest.main()
