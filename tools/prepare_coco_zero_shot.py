#!/usr/bin/env python3
"""Remove supports while preserving an existing COCO SFT query schedule."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwen3vl_sft.data.paths import resolve_media_path
from qwen3vl_sft.data.schema import validate_sft_record
from qwen3vl_sft.evaluation.coco.generation import file_sha256
from qwen3vl_sft.evaluation.coco.protocol import (
    PROMPT_TEMPLATE_VERSION, ZERO_SHOT_TRAIN_SYSTEM_PROMPT, build_question,
)


def convert_record(record, *, data_root, selected_ids):
    if record.get('prompt_template_version') != PROMPT_TEMPLATE_VERSION:
        raise ValueError('source must use the current inst-v5 prompt')
    images = record['image']
    if not isinstance(images, list) or len(images) != record['num_shots'] + 1:
        raise ValueError('expected support images followed by one query image')
    image = resolve_media_path(images[-1], data_root)
    if int(Path(image).stem) not in selected_ids:
        raise ValueError(f'query is outside the previous 10% split: {image}')
    turns = record['conversations']
    if turns[-1]['from'] != 'gpt' or turns[-2]['from'] != 'human':
        raise ValueError('source must end with a query/answer pair')
    result = {
        **record, 'num_shots': 0, 'image': [image],
        'conversations': [
            {'from': 'system', 'value': ZERO_SHOT_TRAIN_SYSTEM_PROMPT},
            {'from': 'human', 'value': '<image>\n' + build_question(record['category'])},
            dict(turns[-1]),
        ],
    }
    validate_sft_record(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='data/coco/train_sft_10pct_1to2to4_11829_inst-v5.json')
    parser.add_argument('--image-ids', default='data/coco/train_image_ids_10pct_124_inst-v5.json')
    parser.add_argument('--data-root', default=str(ROOT / 'data'))
    parser.add_argument('--output', default='data/coco/train_sft_10pct_0shot_11829_inst-v5.json')
    args = parser.parse_args()
    source = json.loads(Path(args.source).read_text())
    split = json.loads(Path(args.image_ids).read_text())
    if split['fraction'] != 0.1 or split['seed'] != source['seed']:
        raise ValueError('source and the 10% split must use the same seed')
    selected_ids = set(split['image_ids'])
    records = [convert_record(r, data_root=args.data_root, selected_ids=selected_ids)
               for r in source['records']]
    payload = {
        **source, 'records': records, 'shots': [0],
        'source_annotations': resolve_media_path(source['source_annotations'], args.data_root),
        'source_manifest': str(Path(args.source).resolve()),
        'source_sha256': file_sha256(args.source),
        'image_ids_manifest': str(Path(args.image_ids).resolve()),
        'image_ids_sha256': file_sha256(args.image_ids),
        'conversion': 'same query images, categories, order and answers; remove all supports',
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f'Wrote {len(records)} zero-shot records; '
          f'{len({r["image"][0] for r in records})} unique query images in the '
          f'{len(split["image_ids"])}-image 10% pool: {output}')


if __name__ == '__main__':
    main()
