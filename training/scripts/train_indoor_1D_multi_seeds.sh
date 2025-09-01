#!/bin/bash
#
# AI8X Indoor Environment Multi-Seed Training Script
# This script runs the indoor environment training with different seeds
# Similar to your model_max_20_seeds.py approach but for ai8x framework
#
# Usage: ./train_indoor_1D_multi_seeds.sh [NUM_SEEDS] [START_SEED]
# Examples:
#   ./train_indoor_1D_multi_seeds.sh           # Default: 20 seeds starting from 42
#   ./train_indoor_1D_multi_seeds.sh 5         # 5 seeds starting from 42 (42-46)
#   ./train_indoor_1D_multi_seeds.sh 10 100    # 10 seeds starting from 100 (100-109)
#

# Parse command line arguments
NUM_SEEDS=${1:-20}        # Default to 20 seeds if not specified
START_SEED=${2:-42}       # Default to starting seed 42 if not specified

echo "========================================="
echo "AI8X Indoor Environment Multi-Seed Training"
echo "========================================="
echo "Number of seeds: $NUM_SEEDS"
echo "Starting seed: $START_SEED"
echo "Seed range: $START_SEED to $((START_SEED + NUM_SEEDS - 1))"
echo "========================================="

# Change to the ai8x-training directory
cd "$(dirname "$0")/.." || exit 1

# Check if the multi-seed Python script exists
if [ ! -f "train_indoor_1D_multi_seeds.py" ]; then
    echo "ERROR: train_indoor_1D_multi_seeds.py not found!"
    echo "Please make sure the multi-seed training script is in the ai8x-training directory."
    exit 1
fi

# Check if required data directory exists
if [ ! -d "data/indoor_environment" ]; then
    echo "ERROR: data/indoor_environment directory not found!"
    echo "Please make sure your indoor environment data is properly set up."
    exit 1
fi

# Check if required model and policy files exist
if [ ! -f "policies/schedule-indoor-env.yaml" ]; then
    echo "WARNING: policies/schedule-indoor-env.yaml not found!"
    echo "The training might fail without the compression policy file."
fi

echo "Starting multi-seed training..."
echo "This will run $NUM_SEEDS training sessions with different seeds ($START_SEED-$((START_SEED + NUM_SEEDS - 1)))"
echo "Results will be saved to ai8x_seed_runs_out/"
echo ""

# Set up PYTHONPATH for distiller (required for ai8x training)
export PYTHONPATH="$PWD/distiller:$PYTHONPATH"
echo "PYTHONPATH set to include distiller module"
echo ""

# Run the multi-seed training with the specified parameters
python train_indoor_1D_multi_seeds.py "$NUM_SEEDS" "$START_SEED"

# Check if the script completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Multi-seed training completed successfully!"
    echo "========================================="
    echo ""
    echo "Results summary:"
    if [ -f "ai8x_seed_runs_out/ai8x_experiment_results_${NUM_SEEDS}_seeds.xlsx" ]; then
        echo "✓ Excel results: ai8x_seed_runs_out/ai8x_experiment_results_${NUM_SEEDS}_seeds.xlsx"
    fi
    if [ -f "ai8x_seed_runs_out/all_runs_results.csv" ]; then
        echo "✓ CSV backup: ai8x_seed_runs_out/all_runs_results.csv"
    fi
    if [ -d "ai8x_seed_runs_out/logs" ]; then
        echo "✓ Individual logs: ai8x_seed_runs_out/logs/"
    fi
    echo ""
    
    # Extract and display mean ± std for test accuracy from CSV
    if [ -f "ai8x_seed_runs_out/all_runs_results.csv" ]; then
        echo "Statistical Summary:"
        python3 -c "
import pandas as pd
import sys
try:
    df = pd.read_csv('ai8x_seed_runs_out/all_runs_results.csv')
    successful_runs = df[df['test_accuracy'] > 0]
    if len(successful_runs) > 0:
        mean_acc = successful_runs['test_accuracy'].mean()
        std_acc = successful_runs['test_accuracy'].std()
        print(f'Test Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}% (n={len(successful_runs)})')
        print(f'Range: {successful_runs[\"test_accuracy\"].min():.2f}% - {successful_runs[\"test_accuracy\"].max():.2f}%')
    else:
        print('No successful runs found.')
except Exception as e:
    print(f'Could not calculate statistics: {e}')
" 2>/dev/null || echo "Could not calculate statistics (pandas not available)"
    fi
    echo ""
    echo "You can now analyze the results to see performance variation across seeds."
else
    echo ""
    echo "========================================="
    echo "Multi-seed training failed!"
    echo "========================================="
    echo ""
    echo "Check the logs in ai8x_seed_runs_out/logs/ for debugging information."
    exit 1
fi
