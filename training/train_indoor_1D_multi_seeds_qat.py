#!/usr/bin/env python3
"""
Indoor Environment Classification — ai8x Multi-Seed QAT Pipeline
- For each seed:
  1) Train with QAT enabled
  2) Quantize best checkpoint to INT8 (q8)
  3) Evaluate the quantized checkpoint (-8 / HW simulation)
  4) Record test accuracy/loss and runtime
"""

import os
import sys
import re
import glob
import time
import shutil
import argparse
import subprocess
from pathlib import Path

import pandas as pd
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='AI8X Multi-Seed QAT Training + Quantize + Evaluate')
    parser.add_argument('num_seeds', nargs='?', type=int, default=5, help='Number of seeds (default: 5)')
    parser.add_argument('start_seed', nargs='?', type=int, default=42, help='Starting seed (default: 42)')
    return parser.parse_args()


args = parse_args()
REPO_ROOT = Path(__file__).resolve().parent

# Config
NUM_REPEATS = args.num_seeds
START_SEED = args.start_seed
SEEDS = [START_SEED + i for i in range(NUM_REPEATS)]
OUTPUT_DIR = REPO_ROOT / "ai8x_seed_runs_out"
LOGS_DIR = OUTPUT_DIR / "logs"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"

OUTPUT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR.mkdir(exist_ok=True)


def build_env():
    env = os.environ.copy()
    distiller_path = str(REPO_ROOT / "distiller")
    env['PYTHONPATH'] = f"{distiller_path}:{env.get('PYTHONPATH','')}"
    # Force GPU 0 if available
    if torch.cuda.is_available():
        env['CUDA_VISIBLE_DEVICES'] = '0'
    return env


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


def run_cmd_tee(cmd_list, log_path, cwd, env):
    cmd = " ".join(cmd_list)
    tee_cmd = f"{cmd} 2>&1 | tee {log_path}"
    return subprocess.run(tee_cmd, shell=True, cwd=str(cwd), env=env, check=True)


print("\n" + "=" * 80)
print("AI8X MULTI-SEED QAT PIPELINE")
print("=" * 80)
try:
    if torch.cuda.is_available():
        ng = torch.cuda.device_count()
        names = ", ".join(torch.cuda.get_device_name(i) for i in range(ng))
        print(f"GPU available: yes ({ng}) -> {names}")
        try:
            torch.cuda.set_device(0)
            print(f"Using GPU 0: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"Warning: failed to set CUDA device 0: {e}")
    else:
        print("GPU available: no (using CPU)")
except Exception as e:
    print(f"GPU check failed: {e}")
print(f"Seeds: {SEEDS}")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 80)


all_results = []

for idx, seed in enumerate(SEEDS, 1):
    run_name = f"indoor_run_1D_seed_{seed}"
    print("\n" + "=" * 80)
    print(f"STARTING RUN {idx}/{NUM_REPEATS} with SEED {seed}")
    print("=" * 80)

    env = build_env()
    start_time = time.time()

    # 1) Train with QAT enabled
    train_log = LOGS_DIR / f"train_run_{idx:02d}_seed_{seed}.log"
    train_cmd = [
        "python", "train.py",
        "--epochs", "10",
        "--batch-size", "256",
        "--optimizer", "Adam",
        "--lr", "0.001",
        "--weight-decay", "0.0005",
        "--use-bias",
        "--deterministic",
        "--model", "ai85indoorenvnetv2",
        "--dataset", "IndoorEnvironment_1D",
        "--data", "data/indoor_environment",
        "--compress", "policies/schedule-indoor-env.yaml",
        "--qat-policy", "policies/qat_policy_indoor.yaml",
        "--device", "MAX78002",
        "--compiler-mode", "none",
        "--out-dir", str(OUTPUT_DIR),
        "--seed", str(seed),
        "--name", run_name,
    ]
    print("Training with QAT...")
    try:
        run_cmd_tee(train_cmd, train_log, REPO_ROOT, env)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: QAT training failed for seed {seed} (code {e.returncode})")
        all_results.append({
            'run': idx, 'seed': seed,
            'test_accuracy_q8': 0.0, 'status': 'train_failed'
        })
        continue

    # Locate best FP32 (QAT) checkpoint (robust search)
    # Primary: runs are typically under OUTPUT_DIR/<run_name_timestamp>/
    run_dirs = sorted(glob.glob(str(OUTPUT_DIR / f"{run_name}*")))
    if not run_dirs:
        print("WARNING: could not locate run directory for checkpoints")
        all_results.append({'run': idx, 'seed': seed, 'test_accuracy_q8': 0.0, 'status': 'no_run_dir'})
        continue
    run_dir = run_dirs[-1]
    best_qat = None
    # Typical names: *_qat_best.pth.tar
    for pat in ["*_qat_best.pth.tar", "*qat_best*.pth.tar", "*best*.pth.tar"]:
        # Search in run_dir
        found = glob.glob(os.path.join(run_dir, pat))
        if found:
            best_qat = sorted(found)[-1]
            break
        # Fallback: recursive search under OUTPUT_DIR
        found = glob.glob(str(OUTPUT_DIR / f"**/{pat}"), recursive=True)
        found = [p for p in found if os.path.basename(p).startswith(run_name)] or found
        if found:
            best_qat = sorted(found)[-1]
            break
    # Last resort: parse train log for a best checkpoint path
    if not best_qat:
        try:
            txt = Path(train_log).read_text()
            m = re.findall(r"=>\s*loading checkpoint\s*(\S*qat_best\S*\.pth\.tar)", txt)
            if m:
                best_qat = m[-1]
        except Exception:
            pass
    if not best_qat:
        print("WARNING: best QAT checkpoint not found")
        all_results.append({'run': idx, 'seed': seed, 'test_accuracy_q8': 0.0, 'status': 'no_best_qat'})
        continue

    # 2) Quantize to INT8 (q8)
    q8_path = os.path.join(run_dir, Path(best_qat).stem + "_q8.pth.tar")
    quant_log = LOGS_DIR / f"quant_run_{idx:02d}_seed_{seed}.log"
    # Ensure quantize.py path (sibling repo: ../ai8x-synthesis)
    QUANTIZE_PY = REPO_ROOT.parent / "ai8x-synthesis" / "quantize.py"
    # Fix q8 naming to avoid double ".pth" in stem
    if best_qat.endswith('.pth.tar'):
        q8_path = best_qat[:-8] + '_q8.pth.tar'
    quant_cmd = [
        "python", str(QUANTIZE_PY),
        best_qat,
        q8_path,
        "--device", "MAX78002",
        "-v",
    ]
    print(f"Quantizing best checkpoint to INT8...\n  best: {best_qat}\n  out : {q8_path}")
    try:
        run_cmd_tee(quant_cmd, quant_log, REPO_ROOT, env)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: quantize failed for seed {seed} (code {e.returncode})")
        all_results.append({'run': idx, 'seed': seed, 'test_accuracy_q8': 0.0, 'status': 'quant_failed'})
        continue

    # 3) Evaluate quantized checkpoint (-8)
    if not os.path.isfile(q8_path):
        print(f"ERROR: q8 file not found after quantization: {q8_path}")
        all_results.append({'run': idx, 'seed': seed, 'test_accuracy_q8': 0.0, 'status': 'q8_missing'})
        continue
    eval_log = LOGS_DIR / f"eval_run_{idx:02d}_seed_{seed}.log"
    eval_cmd = [
        "python", "train.py",
        "--optimizer", "Adam",
        "--model", "ai85indoorenvnetv2",
        "--dataset", "IndoorEnvironment_1D",
        "--data", "data/indoor_environment",
        "--device", "MAX78002",
        "--qat-policy", "policies/qat_policy_indoor.yaml",
        "--use-bias",
        "--deterministic",
        "--weight-decay", "0.0005",
        "--evaluate",
        "--exp-load-weights-from", q8_path,
        "-8",
        "--confusion",
        "--print-freq", "10",
        "--save-sample", "10",
        "--compiler-mode", "none",
        "--out-dir", str(OUTPUT_DIR),
        "--seed", str(seed),
        "--name", run_name + "_q8_eval",
    ]
    print(f"Evaluating INT8 checkpoint...\n  q8: {q8_path}")
    try:
        run_cmd_tee(eval_cmd, eval_log, REPO_ROOT, env)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: evaluation failed for seed {seed} (code {e.returncode})")
        all_results.append({'run': idx, 'seed': seed, 'test_accuracy_q8': 0.0, 'status': 'eval_failed'})
        continue

    # Extract test accuracy from eval log
    test_acc = extract_metric_from_log(
        eval_log,
        [
            r"==>\s*Top1:\s*(\d+\.?\d*)",
            r"Test.*?Top1.*?(\d+\.?\d*)",
            r"Prec@1\s+(\d+\.?\d*)",
            r"Test.*?Accuracy.*?(\d+\.?\d*)",
        ]
    )
    training_time = time.time() - start_time

    all_results.append({
        'run': idx,
        'seed': seed,
        'test_accuracy_q8': test_acc or 0.0,
        'training_time_seconds': training_time,
        'status': 'success' if test_acc is not None else 'no_metric'
    })

    # Copy q8 checkpoint to central dir with seed-prefixed name
    try:
        dest = CHECKPOINTS_DIR / f"{run_name}_{Path(q8_path).name}"
        shutil.copy2(q8_path, dest)
        print(f"Archived q8 checkpoint: {dest}")
    except Exception as e:
        print(f"WARNING: failed to archive q8 checkpoint: {e}")

    # Save rolling results
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_DIR / "all_runs_results_qat.csv", index=False)
    print(f"Completed run {idx}/{NUM_REPEATS} (seed={seed}), q8 Test Acc: {test_acc}")


# Summary
print("\n" + "=" * 80)
print("AGGREGATE RESULTS (QAT q8)")
print("=" * 80)
df = pd.DataFrame(all_results)
df.to_csv(OUTPUT_DIR / "all_runs_results_qat.csv", index=False)
succ = df[df['status'] == 'success']
if len(succ) > 0:
    mean_acc = succ['test_accuracy_q8'].mean()
    std_acc = succ['test_accuracy_q8'].std()
    print(f"SUCCESSFUL RUNS: {len(succ)}/{len(df)}")
    print(f"Mean q8 Test Acc: {mean_acc:.2f}% ± {std_acc:.2f}%")
else:
    print("ERROR: No successful runs!")

print(f"\nResults saved to {OUTPUT_DIR}/")
print("Files created:")
print("- all_runs_results_qat.csv: CSV of q8 results")
print("- logs/: Train/Quant/Eval logs per run")
print("- checkpoints/: q8 checkpoints (archived)")


