#!/usr/bin/env python3
"""
Indoor Environment Classification — ai8x Multi-Seed Training
- Trains 20 times with different seeds using ai8x framework
- Replicates the model_max_20_seeds.py approach but with ai8x train.py
- Collects per-run results and saves summaries
"""

import os
import sys
import json
import pandas as pd
import subprocess
import time
import glob
import re
from pathlib import Path
import argparse
import shutil

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='AI8X Multi-Seed Training')
    parser.add_argument('num_seeds', nargs='?', type=int, default=20, 
                        help='Number of seeds to run (default: 20)')
    parser.add_argument('start_seed', nargs='?', type=int, default=42,
                        help='Starting seed value (default: 42)')
    return parser.parse_args()

args = parse_args()

# Repository root (portable across OS)
REPO_ROOT = Path(__file__).resolve().parent

# Configuration
NUM_REPEATS = args.num_seeds
START_SEED = args.start_seed
SEEDS = [START_SEED + i for i in range(NUM_REPEATS)]
OUTPUT_DIR = "ai8x_seed_runs_out"
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# Storage for results across all runs
all_results = []

# Base training command (similar to train_indoor_1D.sh)
BASE_COMMAND = [
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
    # "--qat-policy", "policies/qat_policy_indoor.yaml",
    "--qat-policy", "None",
    "--device", "MAX78002",
    "--out-dir", OUTPUT_DIR
]

def extract_test_accuracy_from_log(log_file):
    """Extract test accuracy from ai8x training log file."""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Look for test accuracy patterns in ai8x logs
        # Common patterns: "Test Loss: X.XXXX Test Top1: XX.XX"
        patterns = [
            r"Test.*?Top1.*?(\d+\.?\d*)",
            r"Test.*?Accuracy.*?(\d+\.?\d*)",
            r"Final.*?Test.*?(\d+\.?\d*)",
            r"Prec@1\s+(\d+\.?\d*)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                # Return the last (final) accuracy found
                return float(matches[-1])
        
        # If no standard pattern found, look for any percentage near "test"
        test_lines = [line for line in content.lower().split('\n') if 'test' in line and '%' in line]
        if test_lines:
            # Extract the last percentage from the last test line
            last_test_line = test_lines[-1]
            percentages = re.findall(r'(\d+\.?\d*)%', last_test_line)
            if percentages:
                return float(percentages[-1])
                
        return None
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
        return None

def extract_test_loss_from_log(log_file):
    """Extract test loss from ai8x training log file."""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Look for test loss patterns
        patterns = [
            r"Test.*?Loss.*?(\d+\.?\d*)",
            r"Final.*?Loss.*?(\d+\.?\d*)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                return float(matches[-1])
                
        return None
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
        return None

def format_with_precision(df, precision=4):
    """Format DataFrame with specified precision for float values."""
    formatted_df = df.copy()
    for column in formatted_df.select_dtypes(include=['float']):
        formatted_df[column] = formatted_df[column].apply(lambda x: f"{x:.{precision}f}")
    return formatted_df

# Print configuration summary
print(f"\n{'='*80}")
print(f"AI8X MULTI-SEED TRAINING CONFIGURATION")
print(f"{'='*80}")
print(f"Number of runs: {NUM_REPEATS}")
print(f"Starting seed: {START_SEED}")
print(f"Seed range: {START_SEED} to {START_SEED + NUM_REPEATS - 1}")
print(f"Seeds to run: {SEEDS}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"{'='*80}")

# ===== MAIN LOOP - Each iteration runs ai8x training with different seed =====
for run_idx, seed in enumerate(SEEDS):
    print(f"\n{'='*80}")
    print(f"STARTING RUN {run_idx+1}/{NUM_REPEATS} with SEED {seed}")
    print(f"{'='*80}")
    
    # Create unique name for this run
    run_name = f"indoor_run_1D_seed_{seed}"
    log_file = os.path.join(LOGS_DIR, f"run_{run_idx+1:02d}_seed_{seed}.log")
    
    # Build command for this specific run
    command = BASE_COMMAND.copy()
    command.extend([
        "--seed", str(seed),
        "--name", run_name
    ])
    
    print(f"Running command: {' '.join(command)}")
    print(f"Log file: {log_file}")
    
    # Run the training
    start_time = time.time()
    try:
        # Set up environment with proper PYTHONPATH for distiller (portable)
        env = os.environ.copy()
        distiller_path = str(REPO_ROOT / "distiller")
        env['PYTHONPATH'] = f"{distiller_path}:{env.get('PYTHONPATH','')}"
            
        # Run training with real-time output visible in terminal
        # Use tee to both display output and save to log file
        tee_command = f"python {' '.join(command[1:])} 2>&1 | tee {log_file}"
        
        process = subprocess.run(
            tee_command,
            shell=True,
            cwd=str(REPO_ROOT),
            env=env,
            check=True
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.1f} seconds")
        
        # Extract results from log file
        test_accuracy = extract_test_accuracy_from_log(log_file)
        test_loss = extract_test_loss_from_log(log_file)
        
        if test_accuracy is None:
            print(f"WARNING: Could not extract test accuracy from log file")
            test_accuracy = 0.0
        
        if test_loss is None:
            print(f"WARNING: Could not extract test loss from log file")
            test_loss = 0.0
            
        print(f"Extracted - Test Accuracy: {test_accuracy:.2f}%, Test Loss: {test_loss:.4f}")

        # Move/copy best checkpoints into our checkpoints directory
        try:
            # Find the ai8x run directory under OUTPUT_DIR/logs matching this run name
            run_dirs = glob.glob(os.path.join(OUTPUT_DIR, "logs", f"{run_name}*"))
            if run_dirs:
                run_dir = sorted(run_dirs)[-1]
                # Copy any best or checkpoint files into CHECKPOINTS_DIR with seed in name
                for pattern in ["*best*.pth.tar", "*checkpoint*.pth.tar", "*qat_best*.pth.tar", "*qat_checkpoint*.pth.tar"]:
                    for ckpt_path in glob.glob(os.path.join(run_dir, pattern)):
                        dest_name = f"{Path(ckpt_path).name}"
                        # Prefix with run_name to avoid collisions
                        dest_name = f"{run_name}_{dest_name}"
                        dest_path = os.path.join(CHECKPOINTS_DIR, dest_name)
                        shutil.copy2(ckpt_path, dest_path)
                        print(f"Saved checkpoint copy to: {dest_path}")
            else:
                print(f"WARNING: Could not locate run directory under {OUTPUT_DIR}/logs for {run_name}")
        except Exception as e:
            print(f"WARNING: Failed to archive checkpoints for seed {seed}: {e}")
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Training failed for seed {seed}")
        print(f"Return code: {e.returncode}")
        training_time = time.time() - start_time
        test_accuracy = 0.0
        test_loss = 999.0
        
        # Still log the error for debugging
        with open(log_file, 'a') as log_f:
            log_f.write(f"\n\nTRAINING FAILED with return code {e.returncode}\n")
    
    except Exception as e:
        print(f"ERROR: Unexpected error during training for seed {seed}: {e}")
        training_time = time.time() - start_time
        test_accuracy = 0.0
        test_loss = 999.0
    
    # ===== STORE RESULTS FOR THIS RUN =====
    run_result = {
        'run': run_idx + 1,
        'seed': seed,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'training_time_seconds': training_time,
        'run_name': run_name,
        'log_file': log_file
    }
    
    all_results.append(run_result)
    
    # Update Excel file after each run
    current_df = pd.DataFrame(all_results)
    formatted_df = format_with_precision(current_df, precision=4)
    
    # Initialize Excel writer
    excel_path = os.path.join(OUTPUT_DIR, f'ai8x_experiment_results_{NUM_REPEATS}_seeds.xlsx')
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as excel_writer:
        formatted_df.to_excel(excel_writer, sheet_name='Results', index=False)
        
        # Get workbook and worksheet objects
        workbook = excel_writer.book
        worksheet = excel_writer.sheets['Results']
        
        # Add some formatting
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        # Write headers with formatting
        for col_num, value in enumerate(formatted_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
    
    print(f"\nCompleted run {run_idx+1}/{NUM_REPEATS} (seed={seed})")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Results updated in: {excel_path}")

# ===== AGGREGATE RESULTS ACROSS ALL RUNS =====
print(f"\n{'='*80}")
print("AGGREGATE RESULTS ACROSS ALL RUNS")
print(f"{'='*80}")

# Convert to DataFrame
results_df = pd.DataFrame(all_results)

# Calculate statistics for successful runs
successful_runs = results_df[results_df['test_accuracy'] > 0]

# Create summary statistics
if len(successful_runs) > 0:
    summary_stats = {
        'run': 'MEAN',
        'seed': '-',
        'test_accuracy': successful_runs['test_accuracy'].mean(),
        'test_loss': successful_runs['test_loss'].mean(),
        'training_time_seconds': successful_runs['training_time_seconds'].mean(),
        'run_name': f'Statistics (n={len(successful_runs)})',
        'log_file': '-'
    }
    
    std_stats = {
        'run': 'STD',
        'seed': '-',
        'test_accuracy': successful_runs['test_accuracy'].std(),
        'test_loss': successful_runs['test_loss'].std(),
        'training_time_seconds': successful_runs['training_time_seconds'].std(),
        'run_name': f'Standard Deviation',
        'log_file': '-'
    }
    
    min_stats = {
        'run': 'MIN',
        'seed': successful_runs.loc[successful_runs['test_accuracy'].idxmin(), 'seed'],
        'test_accuracy': successful_runs['test_accuracy'].min(),
        'test_loss': successful_runs['test_loss'].max(),  # Higher loss is worse
        'training_time_seconds': successful_runs['training_time_seconds'].min(),
        'run_name': 'Minimum Values',
        'log_file': '-'
    }
    
    max_stats = {
        'run': 'MAX',
        'seed': successful_runs.loc[successful_runs['test_accuracy'].idxmax(), 'seed'],
        'test_accuracy': successful_runs['test_accuracy'].max(),
        'test_loss': successful_runs['test_loss'].min(),  # Lower loss is better
        'training_time_seconds': successful_runs['training_time_seconds'].max(),
        'run_name': 'Maximum Values',
        'log_file': '-'
    }
    
    # Add empty row separator
    separator = {col: '' for col in results_df.columns}
    separator['run'] = '---'
    separator['run_name'] = 'SUMMARY STATISTICS'
    
    # Append summary to results
    results_with_summary = pd.concat([
        results_df,
        pd.DataFrame([separator]),
        pd.DataFrame([summary_stats]),
        pd.DataFrame([std_stats]),
        pd.DataFrame([min_stats]),
        pd.DataFrame([max_stats])
    ], ignore_index=True)
else:
    results_with_summary = results_df

# Save detailed results as CSV backup with summary statistics
results_with_summary.to_csv(os.path.join(OUTPUT_DIR, "all_runs_results.csv"), index=False)

if len(successful_runs) > 0:
    print(f"\nSUCCESSFUL RUNS: {len(successful_runs)}/{len(results_df)}")
    print("\nSUMMARY STATISTICS:")
    print(f"Test Accuracy: {successful_runs['test_accuracy'].mean():.2f}% ± {successful_runs['test_accuracy'].std():.2f}% (n={len(successful_runs)})")
    print(f"Test Loss: {successful_runs['test_loss'].mean():.4f} ± {successful_runs['test_loss'].std():.4f}")
    print(f"Training Time: {successful_runs['training_time_seconds'].mean():.1f}s ± {successful_runs['training_time_seconds'].std():.1f}s")
    
    print(f"\nRANGE:")
    print(f"Best accuracy: {successful_runs['test_accuracy'].max():.2f}% (seed {successful_runs.loc[successful_runs['test_accuracy'].idxmax(), 'seed']})")
    print(f"Worst accuracy: {successful_runs['test_accuracy'].min():.2f}% (seed {successful_runs.loc[successful_runs['test_accuracy'].idxmin(), 'seed']})")
    print(f"Accuracy range: {successful_runs['test_accuracy'].min():.2f}% - {successful_runs['test_accuracy'].max():.2f}%")
else:
    print("ERROR: No successful training runs!")

# Show failed runs if any
failed_runs = results_df[results_df['test_accuracy'] == 0]
if len(failed_runs) > 0:
    print(f"\nFAILED RUNS: {len(failed_runs)}")
    for _, run in failed_runs.iterrows():
        print(f"  Run {run['run']} (seed {run['seed']}): Check log {run['log_file']}")

print(f"\nResults saved to {OUTPUT_DIR}/")
print("Files created:")
print(f"- ai8x_experiment_results_{NUM_REPEATS}_seeds.xlsx: Main results file (updated after each run)")
print(f"- all_runs_results.csv: CSV backup of all results")
print(f"- logs/: Individual log files for each run")
print(f"- checkpoints/: Model checkpoints (if saved by ai8x)")

print(f"\n{'='*80}")
print("ALL RUNS COMPLETED!")
print(f"{'='*80}")

# Print command to run this script
print(f"\nTo run this script:")
print(f"cd {REPO_ROOT}")
print(f"python train_indoor_1D_multi_seeds.py")
