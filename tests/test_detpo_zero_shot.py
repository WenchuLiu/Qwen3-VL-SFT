import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qwen3vl_sft.evaluation.fewshot import cli
from qwen3vl_sft.evaluation.coco.protocol import build_eval_messages


class DetpoZeroShotTests(unittest.TestCase):
    def test_six_dataset_launcher_keeps_detpo_mode_and_pixel_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / 'args.json'
            interpreter = root / 'capture-python'
            interpreter.write_text(
                f'#!{sys.executable}\n'
                'import json, os, sys\n'
                'from pathlib import Path\n'
                'Path(os.environ["CAPTURE_ARGS"]).write_text(json.dumps(sys.argv[1:]))\n'
            )
            interpreter.chmod(0o755)
            env = {key: value for key, value in os.environ.items() if key not in {
                'IE', 'INSTRUCTION_ENHANCEMENT', 'VE', 'VISUAL_ENHANCEMENT',
                'DATASETS', 'CATEGORY_DESCRIPTIONS', 'MAX_IMAGE_PIXELS', 'MAX_PIXELS',
            }}
            env.update(PYTHON_BIN=str(interpreter), CAPTURE_ARGS=str(capture),
                       WORK_ROOT=str(root / 'results'))
            launcher = cli.ROOT / 'scripts/cross_domain_datasets/run_detpo_0shot.sh'
            subprocess.run(['bash', str(launcher)], env=env, check=True, capture_output=True)
            args = json.loads(capture.read_text())
            self.assertIn('--detpo', args)
            self.assertNotIn('--instruction-enhancement', args)
            self.assertEqual(args[args.index('--shots') + 1], '0')
            self.assertEqual(args[args.index('--max-pixels') + 1], '640000')
            self.assertEqual(args[args.index('--datasets') + 1:args.index('--shots')],
                             ['ArTaxOr', 'clipart1k', 'FISH', 'NEU-DET', 'UODD', 'VISUALDIOR'])
            self.assertEqual(Path(args[args.index('--category-descriptions') + 1]),
                             cli.ROOT / 'docs/cross_domain_category_descriptions.json')
            env['IE'] = '1'
            result = subprocess.run(['bash', str(launcher)], env=env, capture_output=True)
            self.assertEqual(result.returncode, 2)

    def test_zero_and_one_shot_select_independent_descriptions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / 'data' / 'FISH'
            (dataset / 'annotations').mkdir(parents=True)
            (dataset / 'test').mkdir()
            (dataset / 'test/query.jpg').touch()
            annotation = {
                'images': [{'id': 1, 'file_name': 'query.jpg', 'width': 100, 'height': 100}],
                'categories': [{'id': 1, 'name': 'fish'}],
                'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1,
                                 'bbox': [10, 10, 20, 20], 'area': 400, 'iscrowd': 0}],
            }
            (dataset / 'annotations/test.json').write_text(json.dumps(annotation))
            descriptions = root / 'descriptions.json'
            descriptions.write_text(json.dumps({'descriptions': {'fish': 'ZERO DESCRIPTION'}}))
            prompt_root = root / 'prompts'
            args = ['eval', '--model-path', 'unused', '--data-root', str(root / 'data'),
                    '--work-dir', str(root / 'out'), '--datasets', 'FISH', '--shots', '0',
                    '--detpo', '--category-descriptions', str(descriptions),
                    '--detpo-prompts', str(prompt_root)]
            observed = []

            def evaluate(**kwargs):
                manifest = json.loads(kwargs['manifest'].read_text())
                record = manifest['records'][0]
                messages = build_eval_messages(record, min_pixels=3136, max_pixels=640000,
                                              detpo=True)
                self.assertEqual(len(messages), 1)
                text = messages[0]['content'][0]['text']
                expected = 'ZERO DESCRIPTION' if record['num_shots'] == 0 else 'ONE DESCRIPTION'
                self.assertIn(expected, text)
                self.assertEqual(len(record['support']), record['num_shots'])
                self.assertEqual(manifest['detpo_prompt_sha256'], kwargs['detpo_prompt_sha256'])
                observed.append(record['num_shots'])
                metric = {'map_50_95': 0, 'map_50': 0, 'map_75': 0}
                return {'metrics_by_shot': {str(record['num_shots']): {
                    'episodes': 1, 'coco_map': {'model': metric, 'ranking': metric}}}}

            with patch.object(sys, 'argv', args), patch.object(cli, 'GenerationRunner') as runner, \
                    patch.object(cli, '_evaluate_one', side_effect=evaluate):
                cli.main()
                self.assertIsNone(runner.call_args.kwargs['category_descriptions'])
            self.assertEqual(observed, [0])
            # Mixed shots must not let the global zero-shot mapping override per-shot prompts.
            (dataset / 'train').mkdir()
            (dataset / 'train/query.jpg').touch()
            (dataset / 'annotations/1_shot.json').write_text(json.dumps(annotation))
            (prompt_root / '1-shot').mkdir(parents=True)
            (prompt_root / '1-shot/all_refined_class_instructions_FISH.json').write_text(
                json.dumps({'fish': 'ONE DESCRIPTION'}))
            args.insert(args.index('--detpo'), '1')
            with patch.object(sys, 'argv', args), patch.object(cli, 'GenerationRunner'), \
                    patch.object(cli, '_evaluate_one', side_effect=evaluate):
                cli.main()
            self.assertEqual(observed, [0, 0, 1])


if __name__ == '__main__':
    unittest.main()
