#!/usr/bin/env python3
"""Run 4-GPU zero-shot SFT, evaluation, GRPO, and matched evaluation in order.

Each stage has its own log and completion marker. Re-running the same command
resumes an interrupted training stage from its most recent Trainer checkpoint.
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def latest_checkpoint(directory):
    checkpoints = [p for p in directory.glob('checkpoint-*')
                   if (p / 'trainer_state.json').is_file()]
    return max(checkpoints, key=lambda p: int(p.name.split('-')[-1])) if checkpoints else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--dataset', type=Path,
                        default=ROOT / 'data/coco/train_sft_10pct_0shot_11829_inst-v5.json')
    parser.add_argument('--episodes', type=Path,
                        default=ROOT / 'data/coco/val_episodes_500_0shot_inst-v5.json')
    parser.add_argument('--smoke', action='store_true', help='two optimizer steps per stage, eight eval tasks')
    parser.add_argument('--grpo-from-base', action='store_true',
                        help='Skip SFT and initialize a fresh LoRA adapter on the original 4B model.')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    with (run / 'pipeline.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute(args, run)
        except Exception as error:
            path = run / 'status.json'
            status = json.loads(path.read_text()) if path.exists() else {}
            status.update(state='failed', error=str(error),
                          updated_at=datetime.now(timezone.utc).isoformat())
            write_json(path, status)
            raise


def execute(args, run):
    model = ROOT / 'weights/Qwen3-VL-4B-Instruct'
    data_root = ROOT / 'data'
    dataset = args.dataset.resolve()
    episodes = args.episodes.resolve()
    config = {
        'model': str(model), 'dataset': str(dataset), 'episodes': str(episodes),
        'smoke': args.smoke, 'cuda_visible_devices': '0,1,2,3',
        'sft_epochs': 3, 'sft_learning_rate': 5e-5, 'lora_r': 64, 'lora_alpha': 128,
        'sft_effective_batch_size': 16, 'grpo_steps': 2 if args.smoke else 1000,
        'grpo_save_steps': 1 if args.smoke else 250, 'grpo_learning_rate': 1e-5,
        'grpo_rollout_prompts': 4, 'grpo_responses_per_prompt': 4,
        'grpo_kl_coef': 0.04, 'grpo_clip_epsilon': 0.2,
        'reward_functions': ['box_iou', 'score', 'format'],
        'min_pixels': 4096, 'max_pixels': 640000, 'seed': 42,
        'eval_max_new_tokens': 1024, 'grpo_max_response_length': 512,
        'sft_coco_eval': 'every epoch', 'grpo_coco_eval_steps': [250, 500, 750, 1000],
        'cross_domain_datasets': ['ArTaxOr', 'clipart1k', 'FISH', 'NEU-DET', 'UODD'],
        'cross_domain_shots': [0], 'cross_domain_eval': 'SFT final, GRPO steps 500 and 1000',
        'metric_definition': 'official COCO AP@[.50:.05:.95]; F1 mean over IoU .50:.05:.95',
    }
    if args.grpo_from_base:
        config.update(
            training_mode='base_model_lora_grpo', initial_adapter_path=None,
            sft_epochs=0, sft_learning_rate=None, sft_effective_batch_size=None,
            sft_coco_eval=None, cross_domain_eval='GRPO steps 500 and 1000',
        )
    title = 'COCO 0-shot inst-v5 Base + LoRA GRPO' if args.grpo_from_base else 'COCO 0-shot inst-v5 SFT + GRPO'
    from hashlib import sha256
    for name, path in [('dataset', dataset), ('episodes', episodes)]:
        config[name + '_sha256'] = sha256(path.read_bytes()).hexdigest()
    if (run / 'config.json').exists():
        if json.loads((run / 'config.json').read_text()) != config:
            raise ValueError('existing run configuration differs; choose a new run directory')
    else:
        write_json(run / 'config.json', config)
    if not (run / 'RESULTS.md').exists():
        (run / 'RESULTS.md').write_text(
            f'# {title}\n\n'
            'Training and evaluation are in progress. Metrics will be written here '
            'after each scheduled evaluation. See status.json for the active stage.\n'
        )
    if args.smoke:
        for name, path, count in [('dataset', dataset, 32), ('episodes', episodes, 8)]:
            payload = json.loads(path.read_text())
            payload['records'] = payload['records'][:count]
            local = run / f'{name}.json'
            write_json(local, payload)
            if name == 'dataset':
                dataset = local
            else:
                episodes = local
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='4',
               TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    for name in ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY',
                 'http_proxy', 'https_proxy', 'all_proxy', 'no_proxy']:
        env.pop(name, None)
    sft, grpo = run / 'sft', run / 'grpo'
    launch = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
              '--nnodes=1', '--nproc_per_node=4', '-m']
    common = ['--model-name-or-path', str(model), '--dataset', str(dataset),
              '--data-root', str(data_root), '--report-to', 'none',
              '--bf16', 'true', '--fp16', 'false', '--tf32', 'true',
              '--lora-enable', 'true', '--lora-r', '64', '--lora-alpha', '128',
              '--attn-implementation', 'sdpa', '--gradient-checkpointing', 'true',
              '--ddp-find-unused-parameters', 'false', '--min-pixels', '4096',
              '--max-pixels', '640000', '--model-max-length', '4096',
              '--eval-mode', 'none', '--eval-strategy', 'no',
              '--seed', '42', '--resume-training', 'false', '--logging-steps', '10']
    def stage(name, command, training_dir=None, target_step=None):
        marker = run / f'{name}.done.json'
        if marker.exists():
            print(f'{name}: already completed', flush=True)
            return
        if training_dir is not None and target_step is not None:
            saved_state = training_dir / f'checkpoint-{target_step}' / 'trainer_state.json'
            if saved_state.is_file():
                state = json.loads(saved_state.read_text())
                if state['global_step'] == target_step:
                    write_json(marker, {'stage': name, 'state': 'completed',
                                        'recovered_from_checkpoint': str(saved_state.parent)})
                    return
        if training_dir is not None:
            checkpoint = latest_checkpoint(training_dir)
            if checkpoint:
                command += ['--resume-from-checkpoint', str(checkpoint)]
        status = {'stage': name, 'state': 'running', 'pid': os.getpid(),
                  'updated_at': datetime.now(timezone.utc).isoformat(), 'command': command}
        write_json(run / 'status.json', status)
        print(f'{name}: starting; log={run / (name + ".log")}', flush=True)
        with (run / f'{name}.log').open('a') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            status['child_pid'] = child.pid
            write_json(run / 'status.json', status)
            code = child.wait()
        status.update(state='completed' if code == 0 else 'failed', exit_code=code,
                      updated_at=datetime.now(timezone.utc).isoformat())
        write_json(run / 'status.json', status)
        if code:
            raise RuntimeError(f'{name} failed ({code}); see {run / (name + ".log")}')
        write_json(marker, status)

    if not args.grpo_from_base:
        stage('sft_train', launch + ['qwen3vl_sft.train'] + common + [
            '--output-dir', str(sft), '--per-device-train-batch-size', '2',
            '--gradient-accumulation-steps', '2', '--learning-rate', '5e-5',
            '--num-train-epochs', '3', '--warmup-ratio', '0.03',
            '--lr-scheduler-type', 'cosine', '--save-strategy', 'epoch',
            '--save-total-limit', '3', '--max-steps', '2' if args.smoke else '-1',
            '--eval-mode', 'generation', '--eval-strategy', 'epoch',
            '--coco-eval-episodes', str(episodes), '--coco-eval-batch-size', '1',
            '--coco-eval-min-pixels', '4096', '--coco-eval-max-pixels', '640000',
            '--coco-eval-max-new-tokens', '1024',
            '--coco-eval-output-dir', str(run / 'sft_coco_eval'),
        ], sft)
        sft_state = json.loads((sft / 'trainer_state.json').read_text())
        if args.smoke:
            if sft_state['global_step'] != 2:
                raise RuntimeError('SFT smoke test did not finish two optimizer steps')
        elif abs(sft_state['epoch'] - 3.0) > 1e-6:
            raise RuntimeError(f'SFT stopped before 3 epochs: {sft_state["epoch"]}')
        epoch_metrics = [r for r in sft_state['log_history'] if 'eval_coco_0shot_map' in r]
        if not args.smoke and len(epoch_metrics) != 3:
            raise RuntimeError('SFT must have a COCO generation evaluation for each of 3 epochs')
        write_json(run / 'sft_epoch_metrics.json', epoch_metrics)

    def evaluate(name, adapter):
        stage(name, [sys.executable, str(ROOT / 'tools/evaluate_coco_parallel.py'),
            '--model-path', str(model), '--adapter-path', str(adapter),
            '--episodes', str(episodes), '--output', str(run / f'{name}.json'),
            '--data-root', str(data_root), '--num-gpus', '4'])

    def cross_domain(name, adapter):
        command = [sys.executable, str(ROOT / 'tools/evaluate_fewshot.py'),
            '--model-path', str(model), '--adapter-path', str(adapter),
            '--data-root', str(data_root), '--work-dir', str(run / name),
            '--datasets', *config['cross_domain_datasets'], '--shots', '0',
            '--num-gpus', '4', '--batch-size', '1', '--min-pixels', '4096',
            '--max-pixels', '640000', '--max-new-tokens', '1024',
            '--attention', 'sdpa', '--seed', '43', '--skip-existing']
        if args.smoke:
            command += ['--max-query-pairs', '2']
        stage(name, command)

    def report():
        metrics = {}
        for name in ['sft_eval', 'grpo_eval_250', 'grpo_eval_500', 'grpo_eval_750', 'grpo_eval']:
            path = run / f'{name}.json'
            if path.exists():
                result = json.loads(path.read_text())['metrics_by_shot']['0']
                key = ('sft' if name == 'sft_eval' else
                       f'grpo_{2 if args.smoke else 1000}' if name == 'grpo_eval' else
                       name.replace('_eval', ''))
                metrics[key] = {
                    'mAP_50_95': result['coco_map']['model']['map_50_95'],
                    'mAP_50': result['coco_map']['model']['map_50'],
                    'mean_f1': result['f1_mean_over_iou'], 'episodes': result['episodes'],
                }
        write_json(run / 'metrics.json', metrics)
        lines = [f'# {title}', '',
                 '| Stage | mAP@0.50:0.95 | Average F1 | Episodes |',
                 '| --- | ---: | ---: | ---: |']
        for name, result in metrics.items():
            lines.append(f'| {name.upper()} | {result["mAP_50_95"]:.6f} | '
                         f'{result["mean_f1"]:.6f} | {result["episodes"]} |')
        cross_metrics = {}
        for name in ['sft_cross_domain', 'grpo_cross_domain_500', 'grpo_cross_domain_1000']:
            directory = run / name
            if not (directory / 'summary.json').is_file():
                continue
            cross_metrics[name] = {}
            for result_file in directory.glob('*/0shot/result.json'):
                result = json.loads(result_file.read_text())['metrics_by_shot']['0']
                cross_metrics[name][result_file.parent.parent.name] = {
                    'mAP_50_95': result['coco_map']['model']['map_50_95'],
                    'mean_f1': result['f1_mean_over_iou'], 'episodes': result['episodes'],
                }
            lines += ['', f'## {name}', '', '| Dataset | mAP@0.50:0.95 | Average F1 | Episodes |',
                      '| --- | ---: | ---: | ---: |']
            for name_dataset, result in cross_metrics[name].items():
                lines.append(f'| {name_dataset} | {result["mAP_50_95"]:.6f} | '
                             f'{result["mean_f1"]:.6f} | {result["episodes"]} |')
        write_json(run / 'cross_domain_metrics.json', cross_metrics)
        averages = {
            name: {key: sum(r[key] for r in results.values()) / len(results)
                   for key in ['mAP_50_95', 'mean_f1']}
            for name, results in cross_metrics.items() if len(results) == 5
        }
        write_json(run / 'cross_domain_macro_average.json', averages)
        if averages:
            lines += ['', '## Equal-weight averages over the five cross-domain datasets', '',
                      '| Stage | Mean mAP | Mean F1 |', '| --- | ---: | ---: |']
            for name, result in averages.items():
                lines.append(f'| {name} | {result["mAP_50_95"]:.6f} | {result["mean_f1"]:.6f} |')
        (run / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
        print(json.dumps(metrics, indent=2), flush=True)

    if not args.grpo_from_base:
        final_sft_eval = run / 'sft_coco_eval' / f'step-{sft_state["global_step"]}_0shot.json'
        write_json(run / 'sft_eval.json', json.loads(final_sft_eval.read_text()))
        report()
        cross_domain('sft_cross_domain', sft)
        report()
    grpo_command = launch + ['qwen3vl_sft.train.grpo'] + common + [
        '--output-dir', str(grpo),
        '--rollout-batch-size', '4', '--rollout-n', '4',
        '--max-prompt-length', '4096', '--max-response-length', '512',
        '--gradient-accumulation-steps', '1', '--learning-rate', '1e-5',
        '--warmup-ratio', '0.03', '--lr-scheduler-type', 'cosine',
        '--max-steps', '2' if args.smoke else '1000',
        '--save-strategy', 'steps', '--save-steps', '1' if args.smoke else '250',
        '--save-total-limit', '4', '--kl-coef', '0.04', '--clip-epsilon', '0.2',
        '--reward-functions', 'box_iou', 'score', 'format',
    ]
    if not args.grpo_from_base:
        grpo_command += ['--adapter-path', str(sft)]
    expected = [1, 2] if args.smoke else [250, 500, 750, 1000]
    for step in expected:
        stage(f'grpo_train_{step}', grpo_command + ['--stop-after-step', str(step)], grpo, step)
        checkpoint = grpo / f'checkpoint-{step}'
        grpo_state = json.loads((checkpoint / 'trainer_state.json').read_text())
        if grpo_state['global_step'] != step:
            raise RuntimeError(f'GRPO did not finish segment at step {step}')
        if not (checkpoint / 'trainer_state.json').is_file():
            raise RuntimeError(f'missing required checkpoint: {checkpoint}')
        name = 'grpo_eval' if step == expected[-1] else f'grpo_eval_{step}'
        evaluate(name, checkpoint)
        report()
        if step % 500 == 0 or (args.smoke and step == expected[-1]):
            cross_domain(f'grpo_cross_domain_{step if not args.smoke else 1000}', checkpoint)
            report()
    write_json(run / 'status.json', {'state': 'completed', 'stage': 'all',
                                   'updated_at': datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__':
    main()
