#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=500:00:00
#SBATCH --job-name=indoor-hawq-p01
#
# Alphas: 101 91  (2 lengths). Run ACE after all five parts finish (see local/hawq_qat_commands_aws.sh).
#   sbatch /shared/b00090279/Indoor-Environment-Quantization/local/slurm_hawq_qat_alpha_part01.sh

set -e

HAWQ_REPO_ROOT="/shared/b00090279/Indoor-Environment-Quantization"
AI8X_TRAINING_ROOT="/shared/b00090279/testMax/ai8x-training"
ENV_PATH="/shared/b00090279/max_env"
HAWQ_DIR="${HAWQ_REPO_ROOT}/hawqv2/hawqv2_runs/indoor_alpha_sweep_pareto_seed42"
HAWQ_CSV="${HAWQ_DIR}/hawq_frontier_candidates_part01.csv"
OUT_TAG="hawq_qat_frontier_p01"

ALPHAS=(101 91)
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
  --out-dir "${OUT_TAG}"

echo "Done ${OUT_TAG}."
