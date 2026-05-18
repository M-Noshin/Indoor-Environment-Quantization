#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=500:00:00
#SBATCH --job-name=parity-one-run
#
# Runs parity_one_run.py (mixed_style vs hawq_style) on one Slurm GPU node, same as:
#   ./parity_one_run.sh [args...]
# Extra arguments after the script name are forwarded to parity_one_run.py (Slurm passes them as $1, $2, ...).
#
# Submit (defaults: L=91, INT 8-8-2-2, seed 42, 10 epochs):
#   sbatch /shared/b00090279/Indoor-Environment-Quantization/local/slurm_parity_one_run.sh
#
# Submit with overrides:
#   sbatch .../slurm_parity_one_run.sh --epochs 2 --in-len 91 --seed 42
#
# Optional env (same as parity_one_run.sh):
#   AI8X_TRAINING_ROOT, AI8X_SYNTHESIS_ROOT

set -euo pipefail

SCRIPT_DIR="/shared/b00090279/Indoor-Environment-Quantization/local"
export AI8X_TRAINING_ROOT="${AI8X_TRAINING_ROOT:-/shared/b00090279/testMax/ai8x-training}"
export AI8X_SYNTHESIS_ROOT="${AI8X_SYNTHESIS_ROOT:-/shared/b00090279/testMax/ai8x-synthesis}"

ENV_PATH="/shared/b00090279/max_env"
source /opt/miniconda/etc/profile.d/conda.sh
conda activate "$ENV_PATH"
[[ "$CONDA_PREFIX" == "$ENV_PATH" ]] || { echo "[NO] conda expected $ENV_PATH got $CONDA_PREFIX"; exit 1; }

ENV_PY="$CONDA_PREFIX/bin/python"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "=== Slurm parity_one_run job ${SLURM_JOB_ID:-?} ==="
echo "Host: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Python: $ENV_PY"
"$ENV_PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())" || true
echo "=== parity_one_run.py ==="

exec "$ENV_PY" -u "${SCRIPT_DIR}/parity_one_run.py" "$@"
