#!/usr/bin/env python3
"""
Multi-seed training for AI8X Indoor Environment 2D model (ai85indoorenvnetv1)
Runs multiple training sessions with different seeds and collects results
Based on train_indoor_1D_multi_seeds.py but for 2D Conv2d model
"""

import os
import sys
import subprocess
import pandas as pd
import argparse
import time
import re
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-seed training for AI8X Indoor Environment 2D')
    parser.add_argument('num_seeds', type=int, default=5, help='Number of seeds to run (default: 5)')
    parser.add_argument('start_seed', type=int, default=42, help='Starting seed value (default: 42)')

    return parser.parse_args()

args = parse_args()

# Configuration
NUM_REPEATS = args.num_seeds
START_SEED = args.start_seed
SEEDS = [START_SEED + i for i in range(NUM_REPEATS)]
OUTPUT_DIR = "ai8x_seed_runs_2D_out"
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# Storage for results across all runs
all_results = []

# Base training command for 2D model (similar to train_indoor.sh)
BASE_COMMAND = [
    "python", "train.py",
    "--epochs", "10",
    "--batch-size", "256",
    "--optimizer", "Adam",
    "--lr", "0.001",
    "--weight-decay", "0.0005",
    "--use-bias",
    "--deterministic",
    "--model", "ai85indoorenvnetv1",        # 2D model
    "--dataset", "IndoorEnvironment",       # 2D dataset  
    "--data", "data/indoor_environment",
    "--compress", "policies/schedule-indoor-env.yaml",
    "--qat-policy", "None",
    "--device", "MAX78002"
]

def extract_test_accuracy_from_log(log_file):
    """Extract test accuracy from ai8x training log file."""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Look for test accuracy patterns
        # Pattern 1: "Top1: XX.XXX%" 
        pattern1 = r'Top1:\s*([\d.]+)%'
        matches1 = re.findall(pattern1, content)
        
        # Pattern 2: "Test Loss: X.XXXX, Test Top1: XX.XXX"
        pattern2 = r'Test Top1:\s*([\d.]+)'
        matches2 = re.findall(pattern2, content)
        
        # Pattern 3: "==> Top1: XX.XXX    Top5: XX.XXX    Loss: X.XXXX"
        pattern3 = r'==> Top1:\s*([\d.]+)'
        matches3 = re.findall(pattern3, content)
        
        # Use the last match found (final test result)
        all_matches = matches1 + matches2 + matches3
        if all_matches:
            return float(all_matches[-1])
        else:
            print(f"⚠️  Could not extract test accuracy from {log_file}")
            return None
            
    except Exception as e:
        print(f"❌ Error reading log file {log_file}: {e}")
        return None

def run_training_with_seed(seed, run_number):
    """Run training with a specific seed"""
    
    # Create unique name for this run
    run_name = f"indoor_run_2D_seed_{seed}"
    log_file = os.path.join(LOGS_DIR, f"run_{run_number:02d}_seed_{seed}.log")
    
    # Build command with seed and name
    command = BASE_COMMAND + [
        "--seed", str(seed),
        "--name", run_name
    ]
    
    print(f"Running command: {' '.join(command)}")
    print(f"Log file: {log_file}")
    
    # Set up environment with PYTHONPATH
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{os.getcwd()}/distiller:{env.get('PYTHONPATH', '')}"
    
    try:
        # Run training with real-time output using tee
        tee_command = f"python train.py --epochs 10 --batch-size 256 --optimizer Adam --lr 0.001 --weight-decay 0.0005 --use-bias --deterministic --model ai85indoorenvnetv1 --dataset IndoorEnvironment --data data/indoor_environment --compress policies/schedule-indoor-env.yaml --qat-policy None --device MAX78002 --seed {seed} --name {run_name} 2>&1 | tee {log_file}"
        
        result = subprocess.run(
            tee_command,
            shell=True,
            env=env,
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            print(f"✅ Training completed successfully for seed {seed}")
            
            # Extract test accuracy from log
            test_accuracy = extract_test_accuracy_from_log(log_file)
            return test_accuracy
        else:
            print(f"❌ Training failed for seed {seed} with return code {result.returncode}")
            return None
            
    except Exception as e:
        print(f"❌ Error running training for seed {seed}: {e}")
        return None

def main():
    """Main execution function"""
    
    print("=" * 80)
    print("AI8X Indoor Environment 2D Multi-Seed Training")
    print("=" * 80)
    print(f"Starting multi-seed training...")
    print(f"This will run {NUM_REPEATS} training sessions with different seeds ({START_SEED}-{START_SEED + NUM_REPEATS - 1})")
    print(f"Model: ai85indoorenvnetv1 (2D Conv2d)")
    print(f"Dataset: IndoorEnvironment (2D)")
    print(f"Results will be saved to {OUTPUT_DIR}/")
    print("PYTHONPATH set to include distiller module")
    
    start_time = time.time()
    
    # Run training for each seed
    for i, seed in enumerate(SEEDS, 1):
        print("\n" + "=" * 80)
        print(f"STARTING RUN {i}/{NUM_REPEATS} with SEED {seed}")
        print("=" * 80)
        
        test_accuracy = run_training_with_seed(seed, i)
        
        # Store results
        result = {
            'run': i,
            'seed': seed,
            'test_accuracy': test_accuracy if test_accuracy is not None else 0.0,
            'status': 'success' if test_accuracy is not None else 'failed'
        }
        all_results.append(result)
        
        # Save intermediate results
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(OUTPUT_DIR, "all_runs_results.csv")
        df.to_csv(csv_path, index=False)
        
        print(f"\n📊 Run {i} completed:")
        print(f"   Seed: {seed}")
        print(f"   Test Accuracy: {test_accuracy:.2f}%" if test_accuracy else "   Test Accuracy: Failed")
        print(f"   Status: {result['status']}")
    
    # Final results processing
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("Multi-seed training completed successfully!")
    print("=" * 80)
    
    # Calculate statistics
    successful_runs = [r for r in all_results if r['status'] == 'success']
    if successful_runs:
        accuracies = [r['test_accuracy'] for r in successful_runs]
        mean_acc = sum(accuracies) / len(accuracies)
        
        if len(accuracies) > 1:
            import statistics
            std_acc = statistics.stdev(accuracies)
            min_acc = min(accuracies)
            max_acc = max(accuracies)
            
            print(f"📊 STATISTICAL RESULTS:")
            print(f"   Successful runs: {len(successful_runs)}/{NUM_REPEATS}")
            print(f"   Test Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
            print(f"   Range: [{min_acc:.2f}%, {max_acc:.2f}%]")
            
            # Add statistics to CSV
            df = pd.DataFrame(all_results)
            
            # Add summary statistics as additional rows
            stats_rows = [
                {'run': 'MEAN', 'seed': '-', 'test_accuracy': mean_acc, 'status': 'summary'},
                {'run': 'STD', 'seed': '-', 'test_accuracy': std_acc, 'status': 'summary'},
                {'run': 'MIN', 'seed': '-', 'test_accuracy': min_acc, 'status': 'summary'},
                {'run': 'MAX', 'seed': '-', 'test_accuracy': max_acc, 'status': 'summary'}
            ]
            
            df_with_stats = pd.concat([df, pd.DataFrame(stats_rows)], ignore_index=True)
            df_with_stats.to_csv(csv_path, index=False)
            
        else:
            print(f"📊 RESULTS:")
            print(f"   Single successful run: {mean_acc:.2f}%")
    else:
        print("❌ No successful runs completed!")
    
    print(f"\n⏱️  Total execution time: {total_time/60:.1f} minutes")
    print(f"\n📁 Results summary:")
    print(f"   ✓ CSV results: {OUTPUT_DIR}/all_runs_results.csv")
    print(f"   ✓ Individual logs: {OUTPUT_DIR}/logs/")
    print(f"   ✓ Model checkpoints: {OUTPUT_DIR}/checkpoints/ (if saved)")

if __name__ == "__main__":
    main()
