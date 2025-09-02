#!/usr/bin/env python3
"""
Quick Mixed-Precision QAT Sanity Sweep (3 configs) for Indoor Environment (1D)
- Runs 3 per-layer weight-bit configs across [8, 4, 2] for 4 layers
- Layers (order): conv1, conv2, fc1, fc2
- For each config and seed:
  1) Generate a per-config QAT policy YAML
  2) Train with QAT
  3) Quantize checkpoint (naming _q{bits} or _qmixed)
  4) Evaluate and aggregate results

Configs:
- INT 8-8-8-8
- INT 8-4-8-8
- INT 4-4-4-4
"""

import os
import sys
import argparse
import itertools
import subprocess
import time
from pathlib import Path
import shutil
import re
import glob
import yaml
import pandas as pd
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Quick Mixed-Precision QAT sweep (2 inputs x 2 configs)')
    parser.add_argument('num_seeds', nargs='?', type=int, default=2,
                        help='Number of seeds per configuration (default: 1)')
    parser.add_argument('start_seed', nargs='?', type=int, default=42,
                        help='Starting seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--device', default='MAX78002')
    parser.add_argument('--data-dir', default='data/indoor_environment')
    parser.add_argument('--out-dir', default=None, help='Override output directory')
    return parser.parse_args()


args = parse_args()
REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = Path(args.out_dir) if args.out_dir else (REPO_ROOT / 'ai8x_seed_runs_out')
LOGS_DIR = OUTPUT_DIR / 'logs_mixed_small'
CHECKPOINTS_DIR = OUTPUT_DIR / 'checkpoints_mixed_small'
POLICY_DIR = REPO_ROOT / 'policies' / 'sweep_small'
OUTPUT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR.mkdir(exist_ok=True)
POLICY_DIR.mkdir(parents=True, exist_ok=True)


def build_env():
    env = os.environ.copy()
    distiller_path = str(REPO_ROOT / 'distiller')
    env['PYTHONPATH'] = f"{distiller_path}:{env.get('PYTHONPATH','')}"
    if torch.cuda.is_available():
        env['CUDA_VISIBLE_DEVICES'] = '0'
    return env


def run_cmd_tee(cmd_list, log_path, cwd, env):
    cmd = ' '.join(cmd_list)
    tee_cmd = f"{cmd} 2>&1 | tee {log_path}"
    return subprocess.run(tee_cmd, shell=True, cwd=str(cwd), env=env, check=True)


def write_rolling_results(all_results, out_dir, results_filename, summary_filename):
    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / results_filename, index=False)
    succ = df[df['status'] == 'success']
    if len(succ) > 0:
        summary = succ.groupby(['input_length', 'config']).agg(
            runs=('test_accuracy', 'count'),
            mean_acc=('test_accuracy', 'mean'),
            std_acc=('test_accuracy', 'std'),
        ).reset_index()
        summary.sort_values(by='mean_acc', ascending=False, inplace=True)
        summary.to_csv(out_dir / summary_filename, index=False)
    else:
        pd.DataFrame(columns=['input_length', 'config', 'runs', 'mean_acc', 'std_acc']).to_csv(
            out_dir / summary_filename, index=False
        )


def write_policy_yaml(path, bits_tuple):
    b1, b2, b3, b4 = bits_tuple
    policy = {
        'start_epoch': 1,
        'weight_bits': 8,
        'shift_quantile': 0.95,
        'outlier_removal_z_score': 2.0,
        'overrides': {
            'conv1': {'weight_bits': int(b1)},
            'conv2': {'weight_bits': int(b2)},
            'fc1':   {'weight_bits': int(b3)},
            'fc2':   {'weight_bits': int(b4)},
        }
    }
    with open(path, 'w') as f:
        yaml.safe_dump(policy, f, sort_keys=False)


def config_name(bits_tuple):
    return f"INT {bits_tuple[0]}-{bits_tuple[1]}-{bits_tuple[2]}-{bits_tuple[3]}"


def detect_weight_bits_label(ckpt_path):
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        sd = ckpt.get('state_dict', {})
        bits_found = set()
        for k, v in sd.items():
            if k.endswith('weight_bits'):
                try:
                    b = int(float(v.detach().cpu().numpy()))
                except Exception:
                    try:
                        b = int(v)
                    except Exception:
                        continue
                if b:
                    bits_found.add(b)
        if len(bits_found) == 1:
            only = next(iter(bits_found))
            return f"INT{only}", f"_q{only}"
        elif len(bits_found) > 1:
            return "mixed", "_qmixed"
    except Exception:
        pass
    return "INT8", "_q8"


def extract_metric_from_log(log_file, patterns):
    try:
        content = Path(log_file).read_text()
    except Exception:
        return None
    for pattern in patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            try:
                return float(matches[-1])
            except Exception:
                continue
    return None


def run_one(in_len, cfg_idx, cfg_bits, cfg_name, seed, policy_file, all_results):
    """Run train->quantize->eval for a single (input_length, config, seed)."""
    run_name = f"indoor_mixed_small_seed_{seed}__L{in_len}__{cfg_bits[0]}_{cfg_bits[1]}_{cfg_bits[2]}_{cfg_bits[3]}"

    # Train
    train_log = LOGS_DIR / f"train_L{in_len}_cfg{cfg_idx:02d}_seed_{seed}.log"
    train_cmd = [
        'python', 'train.py',
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--optimizer', 'Adam',
        '--lr', str(args.lr),
        '--weight-decay', str(args.weight_decay),
        '--use-bias',
        '--deterministic',
        '--model', 'ai85indoorenvnetv2',
        '--dataset', 'IndoorEnvironment_1D',
        '--data', args.data_dir,
        '--input-1d-length', str(in_len),
        '--compress', 'policies/schedule-indoor-env.yaml',
        '--qat-policy', str(policy_file),
        '--device', args.device,
        '--compiler-mode', 'none',
        '--out-dir', str(OUTPUT_DIR),
        '--seed', str(seed),
        '--name', run_name,
    ]
    print(f"Training: {cfg_name} (L={in_len}, seed={seed})")
    try:
        run_cmd_tee(train_cmd, train_log, REPO_ROOT, env)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: training failed (cfg={cfg_name}, seed={seed}) code={e.returncode}")
        all_results.append({'input_length': in_len, 'config': cfg_name, 'seed': seed, 'status': 'train_failed', 'test_accuracy': 0.0})
        write_rolling_results(all_results, OUTPUT_DIR, 'mixed_precision_sweep_small_results.csv', 'mixed_precision_sweep_small_summary.csv')
        return all_results

    # Find best QAT checkpoint
    run_dirs = sorted(glob.glob(str(OUTPUT_DIR / f"{run_name}*")))
    if not run_dirs:
        print("WARNING: run dir not found")
        all_results.append({'input_length': in_len, 'config': cfg_name, 'seed': seed, 'status': 'no_run_dir', 'test_accuracy': 0.0})
        write_rolling_results(all_results, OUTPUT_DIR, 'mixed_precision_sweep_small_results.csv', 'mixed_precision_sweep_small_summary.csv')
        return all_results
    run_dir = run_dirs[-1]
    best_qat = None
    for pat in ["*_qat_best.pth.tar", "*qat_best*.pth.tar", "*best*.pth.tar"]:
        found = glob.glob(os.path.join(run_dir, pat))
        if found:
            best_qat = sorted(found)[-1]
            break
        found = glob.glob(str(OUTPUT_DIR / f"**/{pat}"), recursive=True)
        found = [p for p in found if os.path.basename(p).startswith(run_name)] or found
        if found:
            best_qat = sorted(found)[-1]
            break
    if not best_qat:
        print("WARNING: best QAT checkpoint not found")
        all_results.append({'input_length': in_len, 'config': cfg_name, 'seed': seed, 'status': 'no_best_qat', 'test_accuracy': 0.0})
        write_rolling_results(all_results, OUTPUT_DIR, 'mixed_precision_sweep_small_results.csv', 'mixed_precision_sweep_small_summary.csv')
        return all_results

    # Quantize
    QUANTIZE_PY = REPO_ROOT.parent / 'ai8x-synthesis' / 'quantize.py'
    label, suffix = detect_weight_bits_label(best_qat)
    if best_qat.endswith('.pth.tar'):
        quant_path = best_qat[:-8] + f'{suffix}.pth.tar'
    else:
        quant_path = os.path.join(run_dir, Path(best_qat).stem + f"{suffix}.pth.tar")
    quant_log = LOGS_DIR / f"quant_L{in_len}_cfg{cfg_idx:02d}_seed_{seed}.log"
    quant_cmd = ['python', str(QUANTIZE_PY), best_qat, quant_path, '--device', args.device, '-v']
    print(f"Quantizing ({label}) -> {Path(quant_path).name}")
    try:
        run_cmd_tee(quant_cmd, quant_log, REPO_ROOT, env)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: quantize failed (cfg={cfg_name}, seed={seed}) code={e.returncode}")
        all_results.append({'input_length': in_len, 'config': cfg_name, 'seed': seed, 'status': 'quant_failed', 'test_accuracy': 0.0})
        write_rolling_results(all_results, OUTPUT_DIR, 'mixed_precision_sweep_small_results.csv', 'mixed_precision_sweep_small_summary.csv')
        return all_results

    # Evaluate
    if not os.path.isfile(quant_path):
        print(f"ERROR: quantized file missing: {quant_path}")
        all_results.append({'input_length': in_len, 'config': cfg_name, 'seed': seed, 'status': 'quant_missing', 'test_accuracy': 0.0})
        write_rolling_results(all_results, OUTPUT_DIR, 'mixed_precision_sweep_small_results.csv', 'mixed_precision_sweep_small_summary.csv')
        return all_results
    eval_log = LOGS_DIR / f"eval_L{in_len}_cfg{cfg_idx:02d}_seed_{seed}.log"
    eval_cmd = [
        'python', 'train.py',
        '--deterministic',
        '--optimizer', 'Adam',
        '--model', 'ai85indoorenvnetv2',
        '--dataset', 'IndoorEnvironment_1D',
        '--data', args.data_dir,
        '--input-1d-length', str(in_len),
        '--device', args.device,
        '--qat-policy', str(policy_file),
        '--use-bias',
        '--weight-decay', str(args.weight_decay),
        '--evaluate',
        '--exp-load-weights-from', quant_path,
        '-8',
        '--confusion',
        '--print-freq', '10',
        '--save-sample', '10',
        '--compiler-mode', 'none',
        '--out-dir', str(OUTPUT_DIR),
        '--seed', str(seed),
        '--name', run_name + f"_{label.lower()}_eval",
    ]
    print(f"Evaluating ({label} weights, INT8 activations)")
    try:
        run_cmd_tee(eval_cmd, eval_log, REPO_ROOT, env)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: evaluation failed (cfg={cfg_name}, seed={seed}) code={e.returncode}")
        all_results.append({'input_length': in_len, 'config': cfg_name, 'seed': seed, 'status': 'eval_failed', 'test_accuracy': 0.0})
        write_rolling_results(all_results, OUTPUT_DIR, 'mixed_precision_sweep_small_results.csv', 'mixed_precision_sweep_small_summary.csv')
        return all_results

    test_acc = extract_metric_from_log(
        eval_log,
        [r"==>\s*Top1:\s*(\d+\.?\d*)", r"Test.*?Top1.*?(\d+\.?\d*)",
         r"Prec@1\s+(\d+\.?\d*)", r"Test.*?Accuracy.*?(\d+\.?\d*)"],
    )

    # Archive
    try:
        dest = CHECKPOINTS_DIR / f"{Path(run_name).name}_{Path(quant_path).name}"
        shutil.copy2(quant_path, dest)
    except Exception:
        pass

    elapsed = time.time() - start_time
    all_results.append({
        'input_length': in_len,
        'config': cfg_name,
        'conv1_bits': cfg_bits[0], 'conv2_bits': cfg_bits[1], 'fc1_bits': cfg_bits[2], 'fc2_bits': cfg_bits[3],
        'seed': seed,
        'label_detected': label,
        'test_accuracy': test_acc or 0.0,
        'train_seconds': elapsed,
        'status': 'success' if test_acc is not None else 'no_metric',
    })

    # Rolling write after each run
    write_rolling_results(all_results, OUTPUT_DIR, 'mixed_precision_sweep_small_results.csv', 'mixed_precision_sweep_small_summary.csv')

    return all_results


# Two quick-test configs
PRESET_CONFIGS = [
    (8, 8, 8, 8),
    (4, 4, 4, 4),
]

# Two input lengths to validate pipeline
PRESET_INPUT_LENGTHS = [101, 91]

SEEDS = [args.start_seed + i for i in range(args.num_seeds)]
SEEDS = sorted(SEEDS)

print("\n" + '=' * 80)
print(f"Quick Mixed-Precision Sweep: {len(PRESET_INPUT_LENGTHS)} input lengths x {len(PRESET_CONFIGS)} configs x {len(SEEDS)} seeds")
print('=' * 80)
print(f"Seeds: {SEEDS}")

all_results = []
env = build_env()

for L_idx, in_len in enumerate(PRESET_INPUT_LENGTHS, 1):
    print("\n" + '#' * 80)
    print(f"Input Length {L_idx}/{len(PRESET_INPUT_LENGTHS)}: {in_len}")
    print('#' * 80)

    for cfg_idx, cfg_bits in enumerate(PRESET_CONFIGS, 1):
        cfg_name = config_name(cfg_bits)
        # Base policy template for this input length + config (also write per-run copies per seed)
        base_policy_file = POLICY_DIR / f"qat_policy_L{in_len}_{cfg_bits[0]}_{cfg_bits[1]}_{cfg_bits[2]}_{cfg_bits[3]}.yaml"
        write_policy_yaml(base_policy_file, cfg_bits)

        print("\n" + '-' * 80)
        print(f"Config {cfg_idx}/{len(PRESET_CONFIGS)} | Length {in_len}: {cfg_name}")
        print('-' * 80)

        for s_idx in range(1, args.num_seeds + 1):
            seed = args.start_seed + (s_idx - 1)
            run_name = f"indoor_mixed_small_seed_{seed}__L{in_len}__{cfg_bits[0]}_{cfg_bits[1]}_{cfg_bits[2]}_{cfg_bits[3]}"
            # Per-run policy file to guarantee exact same YAML for train and eval
            policy_file = POLICY_DIR / (
                f"qat_policy_cfg{cfg_idx:02d}_L{in_len}_seed_{seed}_{cfg_bits[0]}_{cfg_bits[1]}_{cfg_bits[2]}_{cfg_bits[3]}.yaml"
            )
            write_policy_yaml(policy_file, cfg_bits)
            start_time = time.time()
            all_results = run_one(in_len, cfg_idx, cfg_bits, cfg_name, seed, policy_file, all_results)

        # All seeds run via run_one; skip legacy block below
        continue
        # Train
        train_log = LOGS_DIR / f"train_L{in_len}_cfg{cfg_idx:02d}_seed_{seed}.log"
        train_cmd = [
            'python', 'train.py',
            '--epochs', str(args.epochs),
            '--batch-size', str(args.batch_size),
            '--optimizer', 'Adam',
            '--lr', str(args.lr),
            '--weight-decay', str(args.weight_decay),
            '--use-bias',
            '--deterministic',
            '--model', 'ai85indoorenvnetv2',
            '--dataset', 'IndoorEnvironment_1D',
            '--data', args.data_dir,
            '--input-1d-length', str(in_len),
            '--compress', 'policies/schedule-indoor-env.yaml',
            '--qat-policy', str(policy_file),
            '--device', args.device,
            '--compiler-mode', 'none',
            '--out-dir', str(OUTPUT_DIR),
            '--seed', str(seed),
            '--name', run_name,
        ]
        print(f"Training: {cfg_name} (L={in_len}, seed={seed})")
        try:
            run_cmd_tee(train_cmd, train_log, REPO_ROOT, env)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: training failed (cfg={cfg_name}, seed={seed}) code={e.returncode}")
            all_results.append({
                'config': cfg_name, 'seed': seed, 'status': 'train_failed', 'test_accuracy': 0.0
            })
            write_rolling_results(
                all_results, OUTPUT_DIR,
                'mixed_precision_sweep_small_results.csv',
                'mixed_precision_sweep_small_summary.csv'
            )
            continue

        # Find best QAT checkpoint
        run_dirs = sorted(glob.glob(str(OUTPUT_DIR / f"{run_name}*")))
        if not run_dirs:
            print("WARNING: run dir not found")
            all_results.append({'config': cfg_name, 'seed': seed, 'status': 'no_run_dir', 'test_accuracy': 0.0})
            write_rolling_results(
                all_results, OUTPUT_DIR,
                'mixed_precision_sweep_small_results.csv',
                'mixed_precision_sweep_small_summary.csv'
            )
            continue
        run_dir = run_dirs[-1]
        best_qat = None
        for pat in ["*_qat_best.pth.tar", "*qat_best*.pth.tar", "*best*.pth.tar"]:
            found = glob.glob(os.path.join(run_dir, pat))
            if found:
                best_qat = sorted(found)[-1]
                break
            found = glob.glob(str(OUTPUT_DIR / f"**/{pat}"), recursive=True)
            found = [p for p in found if os.path.basename(p).startswith(run_name)] or found
            if found:
                best_qat = sorted(found)[-1]
                break
        if not best_qat:
            print("WARNING: best QAT checkpoint not found")
            all_results.append({'config': cfg_name, 'seed': seed, 'status': 'no_best_qat', 'test_accuracy': 0.0})
            write_rolling_results(
                all_results, OUTPUT_DIR,
                'mixed_precision_sweep_small_results.csv',
                'mixed_precision_sweep_small_summary.csv'
            )
            continue

        # Quantize
        QUANTIZE_PY = REPO_ROOT.parent / 'ai8x-synthesis' / 'quantize.py'
        label, suffix = detect_weight_bits_label(best_qat)
        if best_qat.endswith('.pth.tar'):
            quant_path = best_qat[:-8] + f'{suffix}.pth.tar'
        else:
            quant_path = os.path.join(run_dir, Path(best_qat).stem + f"{suffix}.pth.tar")
        quant_log = LOGS_DIR / f"quant_L{in_len}_cfg{cfg_idx:02d}_seed_{seed}.log"
        quant_cmd = ['python', str(QUANTIZE_PY), best_qat, quant_path, '--device', args.device, '-v']
        print(f"Quantizing ({label}) -> {Path(quant_path).name}")
        try:
            run_cmd_tee(quant_cmd, quant_log, REPO_ROOT, env)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: quantize failed (cfg={cfg_name}, seed={seed}) code={e.returncode}")
            all_results.append({'config': cfg_name, 'seed': seed, 'status': 'quant_failed', 'test_accuracy': 0.0})
            write_rolling_results(
                all_results, OUTPUT_DIR,
                'mixed_precision_sweep_small_results.csv',
                'mixed_precision_sweep_small_summary.csv'
            )
            continue

        # Evaluate
        if not os.path.isfile(quant_path):
            print(f"ERROR: quantized file missing: {quant_path}")
            all_results.append({'config': cfg_name, 'seed': seed, 'status': 'quant_missing', 'test_accuracy': 0.0})
            write_rolling_results(
                all_results, OUTPUT_DIR,
                'mixed_precision_sweep_small_results.csv',
                'mixed_precision_sweep_small_summary.csv'
            )
            continue
        eval_log = LOGS_DIR / f"eval_L{in_len}_cfg{cfg_idx:02d}_seed_{seed}.log"
        eval_cmd = [
            'python', 'train.py',
            '--deterministic',
            '--optimizer', 'Adam',
            '--model', 'ai85indoorenvnetv2',
            '--dataset', 'IndoorEnvironment_1D',
            '--data', args.data_dir,
            '--input-1d-length', str(in_len),
            '--device', args.device,
            '--qat-policy', str(policy_file),
            '--use-bias',
            '--weight-decay', str(args.weight_decay),
            '--evaluate',
            '--exp-load-weights-from', quant_path,
            '-8',
            '--confusion',
            '--print-freq', '10',
            '--save-sample', '10',
            '--compiler-mode', 'none',
            '--out-dir', str(OUTPUT_DIR),
            '--seed', str(seed),
            '--name', run_name + f"_{label.lower()}_eval",
        ]
        print(f"Evaluating ({label} weights, INT8 activations)")
        try:
            run_cmd_tee(eval_cmd, eval_log, REPO_ROOT, env)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: evaluation failed (cfg={cfg_name}, seed={seed}) code={e.returncode}")
            all_results.append({'config': cfg_name, 'seed': seed, 'status': 'eval_failed', 'test_accuracy': 0.0})
            write_rolling_results(
                all_results, OUTPUT_DIR,
                'mixed_precision_sweep_small_results.csv',
                'mixed_precision_sweep_small_summary.csv'
            )
            continue

        test_acc = extract_metric_from_log(
            eval_log,
            [r"==>\s*Top1:\s*(\d+\.?\d*)", r"Test.*?Top1.*?(\d+\.?\d*)",
             r"Prec@1\s+(\d+\.?\d*)", r"Test.*?Accuracy.*?(\d+\.?\d*)"],
        )

        # Archive
        try:
            dest = CHECKPOINTS_DIR / f"{Path(run_name).name}_{Path(quant_path).name}"
            shutil.copy2(quant_path, dest)
        except Exception:
            pass

        elapsed = time.time() - start_time
        all_results.append({
            'input_length': in_len,
            'config': cfg_name,
            'conv1_bits': cfg_bits[0], 'conv2_bits': cfg_bits[1], 'fc1_bits': cfg_bits[2], 'fc2_bits': cfg_bits[3],
            'seed': seed,
            'label_detected': label,
            'test_accuracy': test_acc or 0.0,
            'train_seconds': elapsed,
            'status': 'success' if test_acc is not None else 'no_metric',
        })

        # Rolling write after each run
        write_rolling_results(
            all_results, OUTPUT_DIR,
            'mixed_precision_sweep_small_results.csv',
            'mixed_precision_sweep_small_summary.csv'
        )


# Aggregate
df = pd.DataFrame(all_results)
df.to_csv(OUTPUT_DIR / 'mixed_precision_sweep_small_results.csv', index=False)

summary = df[df['status'] == 'success'].groupby(['input_length', 'config']).agg(
    runs=('test_accuracy', 'count'),
    mean_acc=('test_accuracy', 'mean'),
    std_acc=('test_accuracy', 'std'),
).reset_index()
summary.sort_values(by='mean_acc', ascending=False, inplace=True)
summary.to_csv(OUTPUT_DIR / 'mixed_precision_sweep_small_summary.csv', index=False)

print('\n' + '=' * 80)
print('Quick Mixed-Precision Sweep Completed')
print('=' * 80)
print('Summary:')
print(summary.to_string(index=False))
print(f"\nResults saved under: {OUTPUT_DIR}")
print('- mixed_precision_sweep_small_results.csv')
print('- mixed_precision_sweep_small_summary.csv')


