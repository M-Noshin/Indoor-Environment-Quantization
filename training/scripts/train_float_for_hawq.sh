#!/usr/bin/env bash
# Train a float indoor model checkpoint for later HAWQ analysis.

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  ./scripts/train_float_for_hawq.sh [INPUT_LENGTH] [SEED] [OUT_DIR] [RUN_NAME]

Arguments:
  INPUT_LENGTH  1D input length / alpha-specific bandwidth length. Default: 101
  SEED          Random seed. Default: 42
  OUT_DIR       Output directory for logs/checkpoints. Default: hawq_float_runs
  RUN_NAME      Base run name. Default: indoor_float_L<INPUT_LENGTH>_seed_<SEED>

Environment overrides:
  EPOCHS        Default: 10
  BATCH_SIZE    Default: 256
  LR            Default: 0.001
  WEIGHT_DECAY  Default: 0.0005
  DEVICE        Default: MAX78002
  DATA_DIR      Default: data/indoor_environment

Notes:
  - This runs plain float training with --qat-policy None.
  - The output you want for HAWQ is the resulting best.pth.tar checkpoint.
  - Run this from the ai8x-training root after copying the training overlay files.
EOF
  exit 0
fi

INPUT_LENGTH="${1:-101}"
SEED="${2:-42}"
OUT_DIR="${3:-hawq_float_runs}"
RUN_NAME="${4:-indoor_float_L${INPUT_LENGTH}_seed_${SEED}}"

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
DEVICE="${DEVICE:-MAX78002}"
DATA_DIR="${DATA_DIR:-data/indoor_environment}"

if [[ ! -f "train.py" ]]; then
  echo "ERROR: run this script from the ai8x-training root."
  exit 1
fi

mkdir -p "${OUT_DIR}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  if conda env list | awk '{print $1}' | grep -qx "max"; then
    conda activate max
  fi
fi

export PYTHONPATH="$PWD/distiller:${PYTHONPATH:-}"

echo "========================================"
echo "Training float checkpoint for HAWQ"
echo "Input length : ${INPUT_LENGTH}"
echo "Seed         : ${SEED}"
echo "Output dir   : ${OUT_DIR}"
echo "Run name     : ${RUN_NAME}"
echo "Epochs       : ${EPOCHS}"
echo "Batch size   : ${BATCH_SIZE}"
echo "LR           : ${LR}"
echo "Weight decay : ${WEIGHT_DECAY}"
echo "Device       : ${DEVICE}"
echo "Data dir     : ${DATA_DIR}"
echo "========================================"

python train.py \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --optimizer Adam \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --use-bias \
  --deterministic \
  --model ai85indoorenvnetv2 \
  --dataset IndoorEnvironment_1D \
  --data "${DATA_DIR}" \
  --input-1d-length "${INPUT_LENGTH}" \
  --compress policies/schedule-indoor-env.yaml \
  --qat-policy None \
  --device "${DEVICE}" \
  --compiler-mode none \
  --seed "${SEED}" \
  --out-dir "${OUT_DIR}" \
  --name "${RUN_NAME}"

RUN_DIR="$(ls -dt "${OUT_DIR}/${RUN_NAME}"___* 2>/dev/null | grep -v "_eval" | head -1 || true)"
if [[ -z "${RUN_DIR}" ]]; then
  echo "WARNING: could not locate run directory automatically under ${OUT_DIR}"
  exit 0
fi

BEST_CKPT=""
for pat in "*_best.pth.tar" "*best*.pth.tar"; do
  FOUND="$(ls "${RUN_DIR}"/${pat} 2>/dev/null | tail -1 || true)"
  if [[ -n "${FOUND}" ]]; then
    BEST_CKPT="${FOUND}"
    break
  fi
done

echo
echo "Run directory: ${RUN_DIR}"
if [[ -n "${BEST_CKPT}" ]]; then
  echo "Float checkpoint for HAWQ: ${BEST_CKPT}"
else
  echo "WARNING: best checkpoint not found automatically. Inspect ${RUN_DIR}"
fi
