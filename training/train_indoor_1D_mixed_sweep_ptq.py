#!/usr/bin/env python3
"""
Mixed-Precision PTQ Sweep for Indoor Environment (1D)

Key Strategy:
- Train ONE float model per (input_length, seed) → only 5 models for length=101
- For EACH float model, apply all 81 mixed-precision PTQ configs
- Result: 5 seeds × 81 configs = 405 results from just 5 training runs!

Workflow per (length, seed):
  1) Train float model (no QAT, standard training)
  2) For each of 81 bit configs:
     a) Generate QAT policy YAML
     b) Calibrate (PTQ calibration)
     c) Quantize
     d) Evaluate
  3) Aggregate results

Layers: conv1, conv2, fc1, fc2
Bits per layer: [8, 4, 2] → 3^4 = 81 configs
"""

import os
import sys
import itertools
import argparse
import subprocess
import time
from pathlib import Path
import shutil
import re
import glob
import yaml
import pandas as pd
import torch

# Use the same Python interpreter that's running this script
PYTHON_EXECUTABLE = sys.executable


def parse_args():
    parser = argparse.ArgumentParser(description='Mixed-Precision PTQ sweep (train once, PTQ 81 times)')
    parser.add_argument('--num-seeds', type=int, default=5,
                        help='Number of seeds per input length (default: 5)')
    parser.add_argument('--start-seed', type=int, default=42,
                        help='Starting seed (default: 42)')
    parser.add_argument('--input-lengths', type=int, nargs='+', default=[101],
                        help='Input lengths to sweep (default: [101])')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Training epochs for float model')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--calib-batch-size', type=int, default=256,
                        help='Batch size for PTQ calibration')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--z-score', type=float, default=2.0,
                        help='Z-score for outlier removal during PTQ calibration')
    parser.add_argument('--calib-split', default='train',
                        choices=['train', 'val', 'trainval'],
                        help='Data split for PTQ calibration')
    parser.add_argument('--device', default='MAX78002')
    parser.add_argument('--data-dir', default='data/indoor_environment')
    parser.add_argument('--out-dir', default=None,
                        help='Override output directory (default: ptq_mixed_sweep_out)')
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip training, use existing checkpoints')
    return parser.parse_args()


args = parse_args()
REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = Path(args.out_dir) if args.out_dir else (REPO_ROOT / 'ptq_mixed_sweep_out')
LOGS_DIR = OUTPUT_DIR / 'logs'
CHECKPOINTS_DIR = OUTPUT_DIR / 'checkpoints'
POLICY_DIR = OUTPUT_DIR / 'policies'
FLOAT_MODELS_DIR = OUTPUT_DIR / 'float_models'

OUTPUT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR.mkdir(exist_ok=True)
POLICY_DIR.mkdir(exist_ok=True)
FLOAT_MODELS_DIR.mkdir(exist_ok=True)

CALIBRATE_PY = REPO_ROOT / 'calibrate_ptq_simple.py'
QUANTIZE_PY = REPO_ROOT.parent / 'ai8x-synthesis' / 'quantize.py'


def build_env():
    env = os.environ.copy()
    distiller_path = str(REPO_ROOT / 'distiller')
    env['PYTHONPATH'] = f"{distiller_path}:{env.get('PYTHONPATH', '')}"
    if torch.cuda.is_available():
        env['CUDA_VISIBLE_DEVICES'] = '0'
    return env


def run_cmd_tee(cmd_list, log_path, cwd, env):
    """Run command with output tee'd to log file."""
    cmd = ' '.join(cmd_list)
    tee_cmd = f"{cmd} 2>&1 | tee {log_path}"
    return subprocess.run(tee_cmd, shell=True, cwd=str(cwd), env=env, check=True)


def write_policy_yaml(path, bits_tuple, z_score=2.0):
    """Generate QAT policy YAML for mixed-precision PTQ."""
    b1, b2, b3, b4 = bits_tuple
    policy = {
        'start_epoch': 0,  # For evaluation, not training
        'weight_bits': 8,  # default; overridden per layer
        'outlier_removal_z_score': float(z_score),
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
    return f"INT_{bits_tuple[0]}-{bits_tuple[1]}-{bits_tuple[2]}-{bits_tuple[3]}"


def extract_metric_from_log(log_file, patterns):
    """Extract accuracy from log file."""
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


def train_float_model(in_len, seed, all_results):
    """
    Step 1: Train a single floating-point model.
    Returns path to best checkpoint or None on failure.
    """
    run_name = f"indoor_ptq_mixed_float_seed_{seed}_L{in_len}"
    train_log = LOGS_DIR / f"train_float_L{in_len}_seed_{seed}.log"

    train_cmd = [
        PYTHON_EXECUTABLE, 'train.py',
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
        '--device', args.device,
        '--compiler-mode', 'none',
        '--out-dir', str(OUTPUT_DIR),
        '--seed', str(seed),
        '--name', run_name,
    ]

    print(f"\n{'='*80}")
    print(f"Training FLOAT model: L={in_len}, seed={seed}")
    print(f"{'='*80}")

    try:
        run_cmd_tee(train_cmd, train_log, REPO_ROOT, env)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Float training failed (L={in_len}, seed={seed}) code={e.returncode}")
        return None

    run_dirs = sorted(glob.glob(str(OUTPUT_DIR / f"{run_name}*")))
    if not run_dirs:
        print(f"WARNING: No run directory found for {run_name}")
        return None

    run_dir = Path(run_dirs[-1])
    best_ckpt = None
    for pat in ["*best*.pth.tar", "*checkpoint*.pth.tar"]:
        found = list(run_dir.glob(pat))
        if found:
            best_ckpt = sorted(found)[-1]
            break

    if not best_ckpt or not best_ckpt.exists():
        print(f"WARNING: Best checkpoint not found in {run_dir}")
        return None

    dest = FLOAT_MODELS_DIR / f"{run_name}_best.pth.tar"
    shutil.copy2(best_ckpt, dest)
    print(f"✓ Float model saved: {dest.name}")

    return str(dest)


def apply_ptq_config(float_ckpt, in_len, seed, cfg_bits, cfg_name, all_results):
    """
    Steps 2-4: Calibrate → Quantize → Evaluate for one PTQ config.
    """
    start_time = time.time()

    cfg_str = f"{cfg_bits[0]}_{cfg_bits[1]}_{cfg_bits[2]}_{cfg_bits[3]}"
    base_name = f"indoor_ptq_L{in_len}_seed{seed}_{cfg_str}"

    policy_file = POLICY_DIR / f"qat_policy_L{in_len}_seed{seed}_{cfg_str}.yaml"
    write_policy_yaml(policy_file, cfg_bits, args.z_score)

    calib_ckpt = CHECKPOINTS_DIR / f"{base_name}_calibrated.pth.tar"
    calib_log = LOGS_DIR / f"calib_{base_name}.log"

    calib_cmd = [
        PYTHON_EXECUTABLE, str(CALIBRATE_PY),
        '--checkpoint', float_ckpt,
        '--output', str(calib_ckpt),
        '--model', 'ai85indoorenvnetv2',
        '--dataset', 'IndoorEnvironment_1D',
        '--data', args.data_dir,
        '--input-1d-length', str(in_len),
        '--device', args.device,
        '--batch-size', str(args.calib_batch_size),
        '--z-score', str(args.z_score),
        '--calib-split', args.calib_split,
        '--qat-policy', str(policy_file),
    ]

    print(f"  Calibrating {cfg_name}...")
    try:
        run_cmd_tee(calib_cmd, calib_log, REPO_ROOT, env)
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Calibration failed for {cfg_name}, code={e.returncode}")
        all_results.append({
            'input_length': in_len, 'seed': seed, 'config': cfg_name,
            'conv1_bits': cfg_bits[0], 'conv2_bits': cfg_bits[1],
            'fc1_bits': cfg_bits[2], 'fc2_bits': cfg_bits[3],
            'status': 'calib_failed', 'test_accuracy': 0.0,
            'time_seconds': time.time() - start_time
        })
        return all_results

    quant_ckpt = CHECKPOINTS_DIR / f"{base_name}_quantized.pth.tar"
    quant_log = LOGS_DIR / f"quant_{base_name}.log"

    quant_cmd = [
        PYTHON_EXECUTABLE, str(QUANTIZE_PY),
        str(calib_ckpt), str(quant_ckpt),
        '--device', args.device, '-v'
    ]

    print(f"  Quantizing {cfg_name}...")
    try:
        run_cmd_tee(quant_cmd, quant_log, REPO_ROOT, env)
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Quantization failed for {cfg_name}, code={e.returncode}")
        all_results.append({
            'input_length': in_len, 'seed': seed, 'config': cfg_name,
            'conv1_bits': cfg_bits[0], 'conv2_bits': cfg_bits[1],
            'fc1_bits': cfg_bits[2], 'fc2_bits': cfg_bits[3],
            'status': 'quant_failed', 'test_accuracy': 0.0,
            'time_seconds': time.time() - start_time
        })
        return all_results

    eval_log = LOGS_DIR / f"eval_{base_name}.log"

    eval_cmd = [
        PYTHON_EXECUTABLE, 'train.py',
        '--deterministic',
        '--model', 'ai85indoorenvnetv2',
        '--dataset', 'IndoorEnvironment_1D',
        '--data', args.data_dir,
        '--input-1d-length', str(in_len),
        '--device', args.device,
        '--qat-policy', str(policy_file),
        '--use-bias',
        '--evaluate',
        '--exp-load-weights-from', str(quant_ckpt),
        '-8',
        '--print-freq', '10',
        '--compiler-mode', 'none',
        '--out-dir', str(OUTPUT_DIR),
        '--seed', str(seed),
        '--name', f"{base_name}_eval",
    ]

    print(f"  Evaluating {cfg_name}...")
    try:
        run_cmd_tee(eval_cmd, eval_log, REPO_ROOT, env)
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Evaluation failed for {cfg_name}, code={e.returncode}")
        all_results.append({
            'input_length': in_len, 'seed': seed, 'config': cfg_name,
            'conv1_bits': cfg_bits[0], 'conv2_bits': cfg_bits[1],
            'fc1_bits': cfg_bits[2], 'fc2_bits': cfg_bits[3],
            'status': 'eval_failed', 'test_accuracy': 0.0,
            'time_seconds': time.time() - start_time
        })
        return all_results

    test_acc = extract_metric_from_log(
        eval_log,
        [r"==>\s*Top1:\s*(\d+\.?\d*)", r"Test.*?Top1.*?(\d+\.?\d*)",
         r"Prec@1\s+(\d+\.?\d*)", r"Test.*?Accuracy.*?(\d+\.?\d*)"],
    )

    elapsed = time.time() - start_time
    all_results.append({
        'input_length': in_len,
        'seed': seed,
        'config': cfg_name,
        'conv1_bits': cfg_bits[0], 'conv2_bits': cfg_bits[1],
        'fc1_bits': cfg_bits[2], 'fc2_bits': cfg_bits[3],
        'test_accuracy': test_acc or 0.0,
        'time_seconds': elapsed,
        'status': 'success' if test_acc is not None else 'no_metric',
    })

    print(f"  ✓ {cfg_name}: {test_acc:.2f}% (took {elapsed:.1f}s)")

    save_results(all_results)

    return all_results


def save_results(all_results):
    """Save detailed and summary results to CSV."""
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_DIR / 'ptq_mixed_sweep_results.csv', index=False)

    succ = df[df['status'] == 'success']
    if len(succ) > 0:
        summary = succ.groupby(['input_length', 'config']).agg(
            runs=('test_accuracy', 'count'),
            mean_acc=('test_accuracy', 'mean'),
            std_acc=('test_accuracy', 'std'),
            total_time=('time_seconds', 'sum'),
        ).reset_index()
        summary['conv1_bits'] = summary['config'].str.extract(r'INT_(\d+)-')[0].astype(int)
        summary['conv2_bits'] = summary['config'].str.extract(r'-(\d+)-')[0].astype(int)
        summary['fc1_bits'] = summary['config'].str.extract(r'-(\d+)-(\d+)$')[0].astype(int)
        summary['fc2_bits'] = summary['config'].str.extract(r'-(\d+)$')[0].astype(int)
        summary.sort_values(by=['input_length', 'mean_acc'], ascending=[True, False], inplace=True)
        summary.to_csv(OUTPUT_DIR / 'ptq_mixed_sweep_summary.csv', index=False)
    else:
        pd.DataFrame(columns=['input_length', 'config', 'runs', 'mean_acc', 'std_acc']).to_csv(
            OUTPUT_DIR / 'ptq_mixed_sweep_summary.csv', index=False
        )


# ============================================================================
# MAIN EXECUTION
# ============================================================================

BITS = [8, 4, 2]
CONFIGS = list(itertools.product(BITS, repeat=4))  # 81 configs
SEEDS = [args.start_seed + i for i in range(args.num_seeds)]
INPUT_LENGTHS = sorted(args.input_lengths, reverse=True)

print("\n" + '=' * 80)
print(f"Mixed-Precision PTQ Sweep")
print('=' * 80)
print(f"Input lengths: {INPUT_LENGTHS}")
print(f"Seeds: {SEEDS}")
print(f"Bit configs: {len(CONFIGS)} (3^4 combinations of {BITS})")
print(f"Total float models to train: {len(INPUT_LENGTHS)} × {len(SEEDS)} = {len(INPUT_LENGTHS) * len(SEEDS)}")
print(f"Total PTQ evaluations: {len(INPUT_LENGTHS) * len(SEEDS) * len(CONFIGS)} = {len(INPUT_LENGTHS) * len(SEEDS) * len(CONFIGS)}")
print(f"Output: {OUTPUT_DIR}")
print('=' * 80)

env = build_env()
all_results = []

for in_len in INPUT_LENGTHS:
    print(f"\n{'#'*80}")
    print(f"INPUT LENGTH: {in_len}")
    print(f"{'#'*80}")

    for seed in SEEDS:
        print(f"\n{'-'*80}")
        print(f"SEED: {seed} (Length: {in_len})")
        print(f"{'-'*80}")

        if not args.skip_train:
            float_ckpt = train_float_model(in_len, seed, all_results)
        else:
            float_ckpt = FLOAT_MODELS_DIR / f"indoor_ptq_mixed_float_seed_{seed}_L{in_len}_best.pth.tar"
            if not float_ckpt.exists():
                print(f"ERROR: --skip-train specified but float model not found: {float_ckpt}")
                continue
            float_ckpt = str(float_ckpt)
            print(f"Using existing float model: {Path(float_ckpt).name}")

        if not float_ckpt:
            print(f"Skipping all PTQ configs for seed={seed}, L={in_len} (no float model)")
            continue

        print(f"\nApplying {len(CONFIGS)} PTQ configs to seed={seed}, L={in_len}...")
        for cfg_idx, cfg_bits in enumerate(CONFIGS, 1):
            cfg_name = config_name(cfg_bits)
            print(f"\n[{cfg_idx}/{len(CONFIGS)}] {cfg_name}")
            all_results = apply_ptq_config(float_ckpt, in_len, seed, cfg_bits, cfg_name, all_results)

save_results(all_results)

print('\n' + '=' * 80)
print('PTQ Mixed-Precision Sweep COMPLETED!')
print('=' * 80)

df = pd.DataFrame(all_results)
if len(df) > 0:
    summary_df = pd.read_csv(OUTPUT_DIR / 'ptq_mixed_sweep_summary.csv')
    print("\nTop 20 configurations by accuracy:")
    print(summary_df.head(20).to_string(index=False))

    print(f"\n✓ Results saved:")
    print(f"  - {OUTPUT_DIR / 'ptq_mixed_sweep_results.csv'}")
    print(f"  - {OUTPUT_DIR / 'ptq_mixed_sweep_summary.csv'}")
    print(f"  - Checkpoints: {CHECKPOINTS_DIR}")
    print(f"  - Logs: {LOGS_DIR}")
    print(f"  - Float models: {FLOAT_MODELS_DIR}")
else:
    print("\nNo results generated.")

print('=' * 80)
