#!/usr/bin/env bash

eval "$(conda shell.bash hook)"
/shared/b00090279/max_env

export HAWQ_REPO_ROOT=/shared/b00090279/Indoor-Environment-Quantization
export AI8X_TRAINING_ROOT=/shared/b00090279/testMax/ai8x-training

ALPHAS=(101 91 81 71 61 51 41 31 21 11 5)
HAWQ_DIR="${HAWQ_REPO_ROOT}/hawqv2/hawqv2_runs/indoor_alpha_sweep_pareto_seed42"
ITEMS=()
for A in "${ALPHAS[@]}"; do
  ITEMS+=(--item "${A}:${HAWQ_DIR}/results/L${A}.json")
done

cd "${HAWQ_REPO_ROOT}"

python hawqv2/tools/export_hawq_candidates.py \
  "${ITEMS[@]}" \
  --candidate-set frontier \
  --output "${HAWQ_DIR}/hawq_frontier_candidates.csv"

cd "${AI8X_TRAINING_ROOT}"

# Assumes train_indoor_1D_hawq_qat_sweep.py has been copied into ai8x-training.

# Dry run first
python -u train_indoor_1D_hawq_qat_sweep.py \
  --configs-csv "${HAWQ_DIR}/hawq_frontier_candidates.csv" \
  --input-lengths "${ALPHAS[@]}" \
  --num-seeds 5 \
  --start-seed 42 \
  --workers 4 \
  --epochs 10 \
  --z-score 2.0 \
  --out-dir hawq_qat_frontier_out \
  --dry-run

# Full run: remove --dry-run when ready
python -u train_indoor_1D_hawq_qat_sweep.py \
  --configs-csv "${HAWQ_DIR}/hawq_frontier_candidates.csv" \
  --input-lengths "${ALPHAS[@]}" \
  --num-seeds 5 \
  --start-seed 42 \
  --workers 4 \
  --epochs 10 \
  --z-score 2.0 \
  --out-dir hawq_qat_frontier_out

cd "${HAWQ_REPO_ROOT}"

python hawqv2/tools/select_ace_from_hawq.py \
  "${ITEMS[@]}" \
  --eval-csv "${AI8X_TRAINING_ROOT}/hawq_qat_frontier_out/hawq_qat_sweep_summary.csv" \
  --eval-source-label QAT \
  --candidate-set frontier \
  --target-acc 99.2 \
  --beta1 1.0 \
  --beta2 0.0 \
  --limit 20
