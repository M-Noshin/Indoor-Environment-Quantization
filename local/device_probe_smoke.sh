#!/usr/bin/env bash
# Quick smoke: run train.py until DEVICE PROBE + model summary, then exit (no training).
# Matches the early lines of Slurm logs when using --optimizer Adam (optional).
#
# Usage:
#   ./device_probe_smoke.sh
#   MAX_ENV_PYTHON=/path/to/python ./device_probe_smoke.sh
#
# Requires: same deps as full train (conda max_env works).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ROOT="${AI8X_TRAINING_ROOT:-/shared/b00090279/testMax/ai8x-training}"
OUT_DIR="${SCRIPT_DIR}/parity_runs/_device_probe_smoke"
PYTHON_BIN="${MAX_ENV_PYTHON:-/shared/b00090279/max_env/bin/python3}"

export PYTHONPATH="${TRAIN_ROOT}:${TRAIN_ROOT}/distiller:${PYTHONPATH:-}"
mkdir -p "${OUT_DIR}"

cd "${TRAIN_ROOT}"

exec "${PYTHON_BIN}" train.py \
  --device MAX78002 \
  --model ai85indoorenvnetv2 \
  --dataset IndoorEnvironment_1D \
  --data "${TRAIN_ROOT}/data/indoor_environment" \
  --input-1d-length 101 \
  --use-bias \
  --optimizer Adam \
  --lr 0.001 \
  --weight-decay 0.0005 \
  --compress policies/schedule-indoor-env.yaml \
  --qat-policy policies/qat_policy.yaml \
  --compiler-mode none \
  --name device_probe_smoke \
  --out-dir "${OUT_DIR}" \
  --summary model
