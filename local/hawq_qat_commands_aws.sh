#!/usr/bin/env bash
#
# Copy-paste reference only (not meant to run end-to-end as bash).
# If executed: exits immediately below — copy the blocks you need into your shell.
#
# (1) Local single-GPU: export frontier CSV → QAT sweep → ACE on that summary.
# (2) After part dirs hawq_qat_frontier_p01 … p05 exist under AI8X_TRAINING_ROOT
#     (e.g. from cluster Slurm), merge here then ACE on merged summary. Sync dirs first if needed.
# Conda block is optional if python on PATH is already correct.
#
exit 0

export HAWQ_REPO_ROOT=/shared/b00090279/Indoor-Environment-Quantization
export AI8X_TRAINING_ROOT=/shared/b00090279/testMax/ai8x-training
ENV_PATH="${ENV_PATH:-/shared/b00090279/max_env}"

eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"

ALPHAS=(101 91 81 71 61 51 41 31 21 11 5)
HAWQ_DIR="${HAWQ_REPO_ROOT}/hawqv2/hawqv2_runs/indoor_alpha_sweep_pareto_seed42"
ITEMS=()
for A in "${ALPHAS[@]}"; do
  ITEMS+=(--item "${A}:${HAWQ_DIR}/results/L${A}.json")
done

# =============================================================================
# (1) Single-GPU path: one frontier CSV, one QAT out dir, ACE on that summary
#     ACE: add --report-csv to persist ranked candidates (otherwise mostly terminal output).
# =============================================================================

cd "${HAWQ_REPO_ROOT}"

python -u hawqv2/tools/export_hawq_candidates.py \
  "${ITEMS[@]}" \
  --candidate-set frontier \
  --output "${HAWQ_DIR}/hawq_frontier_candidates.csv"

cp -f "${HAWQ_REPO_ROOT}/training/train_indoor_1D_hawq_qat_sweep.py" "${AI8X_TRAINING_ROOT}/"

cd "${AI8X_TRAINING_ROOT}"

# Optional dry-run (instead of the full train block below):
# python -u train_indoor_1D_hawq_qat_sweep.py \
#   --configs-csv "${HAWQ_DIR}/hawq_frontier_candidates.csv" \
#   --input-lengths "${ALPHAS[@]}" \
#   --num-seeds 5 --start-seed 42 --workers 4 --epochs 10 --z-score 2.0 \
#   --out-dir hawq_qat_frontier_out --dry-run

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

python -u hawqv2/tools/select_ace_from_hawq.py \
  "${ITEMS[@]}" \
  --eval-csv "${AI8X_TRAINING_ROOT}/hawq_qat_frontier_out/hawq_qat_sweep_summary.csv" \
  --eval-source-label QAT \
  --candidate-set frontier \
  --target-acc 99.2 \
  --beta1 1.0 \
  --beta2 0.0 \
  --limit 20 \
  --report-csv "${AI8X_TRAINING_ROOT}/hawq_qat_frontier_out/hawq_frontier_ace_report.csv"

# =============================================================================
# (2) Merge part CSVs → hawq_qat_frontier_merged/ (re-run when more parts finish)
#     Outputs: hawq_qat_sweep_results.csv, hawq_qat_sweep_summary.csv (use summary for ACE)
# =============================================================================

python "${HAWQ_REPO_ROOT}/local/merge_hawq_qat_parallel_parts.py" \
  --ai8x-training-root "${AI8X_TRAINING_ROOT}" \
  --parts 01,02,03,04,05 \
  --part-prefix hawq_qat_frontier_p \
  --merged-dir-name hawq_qat_frontier_merged

# =============================================================================
# (3) ACE on merged QAT summary (needs ITEMS from the env block above; cwd = repo root)
#     Expect joined: N / 191 until all parts complete; missing eval rows shrinks as N grows.
#     --report-csv writes ACE-ranked rows (still prints a short summary to the terminal).
# =============================================================================

cd "${HAWQ_REPO_ROOT}"

python -u hawqv2/tools/select_ace_from_hawq.py \
  "${ITEMS[@]}" \
  --eval-csv "${AI8X_TRAINING_ROOT}/hawq_qat_frontier_merged/hawq_qat_sweep_summary.csv" \
  --eval-source-label QAT \
  --candidate-set frontier \
  --target-acc 99.2 \
  --beta1 1.0 \
  --beta2 0.0 \
  --limit 20 \
  --report-csv "${AI8X_TRAINING_ROOT}/hawq_qat_frontier_merged/hawq_frontier_ace_report.csv"
