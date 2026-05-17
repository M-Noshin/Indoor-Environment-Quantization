#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=500:00:00
#SBATCH --job-name=indoor-hawq-qat-1gpu
#
# Single-GPU: all 11 input lengths, no ACE (run ACE after training).
# For 5 GPUs in parallel use instead:
#   local/slurm_hawq_qat_alpha_part01.sh … part05.sh
#
# Submit:
#   sbatch /shared/b00090279/Indoor-Environment-Quantization/local/slurm_indoor_mixed_qat_full_sweep.sh

set -e

HAWQ_REPO_ROOT="/shared/b00090279/Indoor-Environment-Quantization"
AI8X_TRAINING_ROOT="/shared/b00090279/testMax/ai8x-training"
ENV_PATH="/shared/b00090279/max_env"

ALPHAS=(101 91 81 71 61 51 41 31 21 11 5)
HAWQ_DIR="${HAWQ_REPO_ROOT}/hawqv2/hawqv2_runs/indoor_alpha_sweep_pareto_seed42"
HAWQ_CSV="${HAWQ_DIR}/hawq_frontier_candidates.csv"
ITEMS=()
for A in "${ALPHAS[@]}"; do
  ITEMS+=(--item "${A}:${HAWQ_DIR}/results/L${A}.json")
done

source /opt/miniconda/etc/profile.d/conda.sh
conda activate "$ENV_PATH"
[[ "$CONDA_PREFIX" == "$ENV_PATH" ]] || { echo "[NO] conda"; exit 1; }

ENV_PY="$CONDA_PREFIX/bin/python"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "${HAWQ_REPO_ROOT}"
"$ENV_PY" -u hawqv2/tools/export_hawq_candidates.py \
  "${ITEMS[@]}" \
  --candidate-set frontier \
  --output "${HAWQ_CSV}"

cp -f "${HAWQ_REPO_ROOT}/training/train_indoor_1D_hawq_qat_sweep.py" "${AI8X_TRAINING_ROOT}/"

cd "${AI8X_TRAINING_ROOT}"
"$ENV_PY" -u train_indoor_1D_hawq_qat_sweep.py \
  --configs-csv "${HAWQ_CSV}" \
  --input-lengths "${ALPHAS[@]}" \
  --num-seeds 5 \
  --start-seed 42 \
  --workers 4 \
  --epochs 10 \
  --z-score 2.0 \
  --out-dir hawq_qat_frontier_out

echo "Done indoor-hawq-qat-1gpu (no ACE)."
