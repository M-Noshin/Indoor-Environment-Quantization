#!/bin/bash

# Multi-seed training script for AI8X Indoor Environment 2D model
# Usage: ./scripts/train_indoor_2D_multi_seeds.sh [NUM_SEEDS] [START_SEED]
# Example: ./scripts/train_indoor_2D_multi_seeds.sh 5 42

# Default values
NUM_SEEDS=${1:-5}      # Number of seeds to run (default: 5)
START_SEED=${2:-42}    # Starting seed value (default: 42)

echo "========================================="
echo "AI8X Indoor Environment 2D Multi-Seed Training"
echo "========================================="
echo "Model: ai85indoorenvnetv1 (2D Conv2d)"
echo "Dataset: IndoorEnvironment (2D)"
echo "Number of seeds: $NUM_SEEDS"
echo "Starting seed: $START_SEED"
echo "Output directory: ai8x_seed_runs_2D_out/"
echo ""

# Set up environment
export PYTHONPATH="$PWD/distiller:$PYTHONPATH"

# Check if distiller is available
if [ ! -d "distiller" ]; then
    echo "❌ Error: distiller directory not found!"
    echo "   Make sure you're running from the ai8x-training root directory"
    exit 1
fi

# Check if the 2D model file exists
if [ ! -f "models/ai85net_indoor_env_v1.py" ]; then
    echo "❌ Error: 2D model file not found!"
    echo "   Expected: models/ai85net_indoor_env_v1.py"
    exit 1
fi

# Check if the 2D dataset file exists  
if [ ! -f "datasets/indoor_environment.py" ]; then
    echo "❌ Error: 2D dataset file not found!"
    echo "   Expected: datasets/indoor_environment.py"
    exit 1
fi

# Check if data directory exists
if [ ! -d "data/indoor_environment" ]; then
    echo "❌ Error: Data directory not found!"
    echo "   Expected: data/indoor_environment/"
    exit 1
fi

echo "✅ Environment checks passed"
echo ""

# Run the Python multi-seed training script
echo "🚀 Starting multi-seed training..."
python train_indoor_2D_multi_seeds.py $NUM_SEEDS $START_SEED

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ Multi-seed training completed!"
    echo "========================================="
    
    # Display summary statistics if available
    CSV_FILE="ai8x_seed_runs_2D_out/all_runs_results.csv"
    if [ -f "$CSV_FILE" ]; then
        echo ""
        echo "📊 FINAL STATISTICS:"
        echo "-------------------"
        
        # Use Python to calculate and display statistics
        python -c "
import pandas as pd
import sys

try:
    df = pd.read_csv('$CSV_FILE')
    successful_runs = df[df['status'] == 'success']
    
    if len(successful_runs) > 0:
        accuracies = successful_runs['test_accuracy'].values
        mean_acc = accuracies.mean()
        
        if len(accuracies) > 1:
            std_acc = accuracies.std()
            min_acc = accuracies.min()
            max_acc = accuracies.max()
            
            print(f'Successful runs: {len(successful_runs)}/{len(df[df[\"status\"] != \"summary\"])}')
            print(f'Test Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%')
            print(f'Range: [{min_acc:.2f}%, {max_acc:.2f}%]')
        else:
            print(f'Single run accuracy: {mean_acc:.2f}%')
    else:
        print('No successful runs found!')
        
except Exception as e:
    print(f'Could not read results: {e}')
" 2>/dev/null
        
        echo ""
        echo "📁 Results saved to:"
        echo "   • CSV file: $CSV_FILE"
        echo "   • Log files: ai8x_seed_runs_2D_out/logs/"
        echo "   • Checkpoints: ai8x_seed_runs_2D_out/checkpoints/"
    fi
    
else
    echo ""
    echo "========================================="
    echo "❌ Multi-seed training failed!"
    echo "========================================="
    echo "Check the logs in ai8x_seed_runs_2D_out/logs/ for details"
    exit 1
fi

echo ""
echo "🎯 To analyze results further:"
echo "   python -c \"import pandas as pd; df=pd.read_csv('ai8x_seed_runs_2D_out/all_runs_results.csv'); print(df)\""
echo ""
echo "🔄 To run again with different parameters:"
echo "   ./scripts/train_indoor_2D_multi_seeds.sh [NUM_SEEDS] [START_SEED]"
echo "   Example: ./scripts/train_indoor_2D_multi_seeds.sh 10 100"
