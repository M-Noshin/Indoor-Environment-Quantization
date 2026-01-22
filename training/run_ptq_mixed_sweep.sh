#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=500:00:00
#SBATCH --job-name=ptq_mixed_sweep
#SBATCH --output=ptq_mixed_sweep_%j.out
#SBATCH --error=ptq_mixed_sweep_%j.err

# Initialize conda
source ~/.bashrc

# Activate environment
conda activate /shared/b00090279/max

# Lock in the interpreter
ENV_PY="$CONDA_PREFIX/bin/python"

# Verify activation
if [[ "$CONDA_DEFAULT_ENV" == "/shared/b00090279/max" || "$CONDA_DEFAULT_ENV" == "b00090279" ]]; then
    echo "[YES] Conda environment activated: $CONDA_DEFAULT_ENV"
else
    echo "[NO] Conda environment did NOT activate."
    echo "Current python: $(which python)"
    exit 1
fi

echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "Using Python: $ENV_PY"

# Sanity check
"$ENV_PY" -c "import sys, torch, os; \
print('sys.executable:', sys.executable); \
print('CONDA_PREFIX:', os.environ.get('CONDA_PREFIX')); \
print('torch:', torch.__version__); \
print('cuda available:', torch.cuda.is_available())"

# ========== NAVIGATE TO THE CORRECT DIRECTORY ==========
cd /shared/b00090279/testMax/ai8x-training || { echo "Failed to cd to ai8x-training"; exit 1; }
echo "Working directory: $PWD"

# Set PYTHONPATH relative to current directory
export PYTHONPATH="$PWD/distiller:${PYTHONPATH:-}"

# Run the PTQ mixed-precision sweep
# - 5 seeds (42-46)
# - Input length: 101
# - 81 bit configs (8/4/2 per layer)
# - Result: 405 PTQ evaluations from just 5 training runs!
echo ""
echo "=========================================="
echo "Starting PTQ Mixed-Precision Sweep"
echo "=========================================="
echo "Seeds: 5 (42-46)"
echo "Input lengths: 101"
echo "Bit configs: 81 (3^4 combinations)"
echo "Total models to train: 5"
echo "Total PTQ evaluations: 405"
echo "Output: $PWD/ptq_mixed_sweep_out/"
echo "=========================================="
echo ""

"$ENV_PY" -u train_indoor_1D_mixed_sweep_ptq.py \
  --num-seeds 5 \
  --start-seed 42 \
  --input-lengths 101 \
  --epochs 10 \
  --z-score 2.0 \
  --calib-split train

echo ""
echo "=========================================="
echo "PTQ Mixed-Precision Sweep Complete!"
echo "=========================================="
echo "Results: $PWD/ptq_mixed_sweep_out/ptq_mixed_sweep_summary.csv"
echo "=========================================="
