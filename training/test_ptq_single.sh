#!/bin/bash
# Pure PTQ Test for Indoor Environment Model
# Flow: Train (no QAT) → Calibrate → Quantize → Evaluate
#
# Expected Results:
#   - Float model (10 epochs): ~99.7-99.9%
#   - PTQ INT8 quantized: ~97-98%
#   - QAT INT8 quantized: ~98-99%
#   - PTQ gap vs QAT: ~0.5-1% (expected and acceptable)

set -e  # Exit on error

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate max

# Setup Python path for distiller (handle unbound variable)
export PYTHONPATH=$PWD/distiller:${PYTHONPATH:-}

# Configuration
SEED=42
INPUT_LENGTH=101
EPOCHS=10  # Train normally (no QAT) - same as QAT for fair comparison
BATCH_SIZE=256
LR=0.001
WEIGHT_DECAY=0.0005
DEVICE="MAX78002"
DATA_DIR="data/indoor_environment"
OUT_DIR="ptq_test_output"
RUN_NAME="indoor_ptq_test_seed_${SEED}_L${INPUT_LENGTH}"

# Mixed precision config to test
CONV1_BITS=4
CONV2_BITS=4
FC1_BITS=4
FC2_BITS=4
CONFIG_NAME="INT_${CONV1_BITS}-${CONV2_BITS}-${FC1_BITS}-${FC2_BITS}"

# PTQ knobs:
# - calibration split: 'trainval' provides better coverage for activation statistics
# - clipping method: Leave empty to use MAX_BIT_SHIFT (default), which works with calibrated thresholds
#   NOTE: --clip-method SCALE overrides calibrated thresholds and performs worse!
CALIB_SPLIT="${CALIB_SPLIT:-trainval}"         # train | trainval
PTQ_CLIP_METHOD="${PTQ_CLIP_METHOD:-}"         # Leave empty for MAX_BIT_SHIFT (default, works best with calibration)
PTQ_SCALE="${PTQ_SCALE:-1.0}"                  # only used when PTQ_CLIP_METHOD=SCALE

echo "========================================"
echo "Pure PTQ Test - ${CONFIG_NAME}"
echo "Input Length: ${INPUT_LENGTH}, Seed: ${SEED}"
echo "Conv1: ${CONV1_BITS}-bit, Conv2: ${CONV2_BITS}-bit, FC1: ${FC1_BITS}-bit, FC2: ${FC2_BITS}-bit"
echo "========================================"

# Create QAT policy for mixed-precision quantization
QAT_POLICY_FILE="${OUT_DIR}/qat_policy_${CONFIG_NAME}.yaml"
mkdir -p "${OUT_DIR}"
cat > "${QAT_POLICY_FILE}" <<EOF
---
start_epoch: 0
weight_bits: 8
outlier_removal_z_score: 2.0
overrides:
  conv1:
    weight_bits: ${CONV1_BITS}
  conv2:
    weight_bits: ${CONV2_BITS}
  fc1:
    weight_bits: ${FC1_BITS}
  fc2:
    weight_bits: ${FC2_BITS}
EOF
echo "Created QAT policy: ${QAT_POLICY_FILE}"
cat "${QAT_POLICY_FILE}"
echo ""
echo "Flow: Train (no QAT) → Calibrate (with policy) → Quantize (reads bits from checkpoint) → Evaluate"
echo ""

# Step 1: Train WITHOUT QAT
echo "Step 1/4: Training (no QAT)..."
python train.py \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --optimizer Adam \
  --lr ${LR} \
  --weight-decay ${WEIGHT_DECAY} \
  --use-bias \
  --deterministic \
  --model ai85indoorenvnetv2 \
  --dataset IndoorEnvironment_1D \
  --data ${DATA_DIR} \
  --input-1d-length ${INPUT_LENGTH} \
  --compress policies/schedule-indoor-env.yaml \
  --qat-policy None \
  --device ${DEVICE} \
  --compiler-mode none \
  --out-dir ${OUT_DIR} \
  --seed ${SEED} \
  --name ${RUN_NAME} \
  2>&1 | tee ${OUT_DIR}/train_${RUN_NAME}.log

# Find the run directory (exclude eval runs)
RUN_DIR=$(ls -dt ${OUT_DIR}/${RUN_NAME}___* 2>/dev/null | grep -v "_eval" | head -1)
if [ -z "${RUN_DIR}" ]; then
    echo "ERROR: Training run directory not found"
    echo "Looking for: ${OUT_DIR}/${RUN_NAME}___*"
    ls -dt ${OUT_DIR}/${RUN_NAME}* 2>/dev/null || echo "No directories found"
    exit 1
fi
echo ""
echo "Run directory: ${RUN_DIR}"

# Find best checkpoint
BEST_CKPT=""
for pat in "*_best.pth.tar" "*best*.pth.tar"; do
    FOUND=$(ls ${RUN_DIR}/${pat} 2>/dev/null | tail -1)
    if [ -n "${FOUND}" ]; then
        BEST_CKPT="${FOUND}"
        break
    fi
done

if [ -z "${BEST_CKPT}" ]; then
    echo "ERROR: Best checkpoint not found in ${RUN_DIR}"
    ls -la ${RUN_DIR}/ 2>/dev/null | head -20
    exit 1
fi
echo "Best checkpoint: ${BEST_CKPT}"

# Step 2: Calibrate (fuses BN, collects activation stats, sets activation_threshold and final_scale)
echo ""
echo "Step 2/4: Calibrating (BN fusion + activation statistics)..."
CALIBRATED_CKPT="${RUN_DIR}/${RUN_NAME}_calibrated.pth.tar"
python calibrate_ptq_simple.py \
  --checkpoint ${BEST_CKPT} \
  --output ${CALIBRATED_CKPT} \
  --model ai85indoorenvnetv2 \
  --dataset IndoorEnvironment_1D \
  --data ${DATA_DIR} \
  --input-1d-length ${INPUT_LENGTH} \
  --device ${DEVICE} \
  --batch-size ${BATCH_SIZE} \
  --z-score 2.0 \
  --calib-split ${CALIB_SPLIT} \
  --qat-policy ${QAT_POLICY_FILE} \
  2>&1 | tee ${RUN_DIR}/calibrate_${RUN_NAME}.log

if [ ! -f "${CALIBRATED_CKPT}" ]; then
    echo "ERROR: Calibrated checkpoint not found at ${CALIBRATED_CKPT}"
    exit 1
fi
echo "Calibrated checkpoint: ${CALIBRATED_CKPT}"

# Step 3: Quantize
echo ""
echo "Step 3/4: Quantizing to ${CONFIG_NAME}..."
QUANT_CKPT="${CALIBRATED_CKPT%.pth.tar}_q.pth.tar"

# Build quantize args - empty PTQ_CLIP_METHOD means use default MAX_BIT_SHIFT
QUANTIZE_ARGS=""
if [ -n "${PTQ_CLIP_METHOD}" ]; then
  echo "Using custom clip method: ${PTQ_CLIP_METHOD}"
  QUANTIZE_ARGS="--clip-method ${PTQ_CLIP_METHOD}"
  if [ "${PTQ_CLIP_METHOD}" = "SCALE" ]; then
    QUANTIZE_ARGS="${QUANTIZE_ARGS} --scale ${PTQ_SCALE}"
  fi
else
  echo "Using default MAX_BIT_SHIFT (respects calibrated thresholds)"
fi

python ../ai8x-synthesis/quantize.py \
  ${CALIBRATED_CKPT} \
  ${QUANT_CKPT} \
  --device ${DEVICE} \
  ${QUANTIZE_ARGS} \
  -v \
  2>&1 | tee ${RUN_DIR}/quant_${RUN_NAME}.log

if [ ! -f "${QUANT_CKPT}" ]; then
    echo "ERROR: Quantized checkpoint not found at ${QUANT_CKPT}"
    exit 1
fi
echo "Quantized checkpoint: ${QUANT_CKPT}"

# Step 4: Evaluate
echo ""
echo "Step 4/4: Evaluating quantized model..."

# Use the same QAT policy for evaluation (signals BN is fused + mixed-precision bits)
# (We already created this at the start of the script)

python train.py \
  --deterministic \
  --optimizer Adam \
  --model ai85indoorenvnetv2 \
  --dataset IndoorEnvironment_1D \
  --data ${DATA_DIR} \
  --input-1d-length ${INPUT_LENGTH} \
  --device ${DEVICE} \
  --qat-policy ${QAT_POLICY_FILE} \
  --use-bias \
  --weight-decay ${WEIGHT_DECAY} \
  --evaluate \
  --exp-load-weights-from ${QUANT_CKPT} \
  -8 \
  --confusion \
  --print-freq 10 \
  --compiler-mode none \
  --out-dir ${OUT_DIR} \
  --seed ${SEED} \
  --name ${RUN_NAME}_eval \
  2>&1 | tee ${RUN_DIR}/eval_${RUN_NAME}.log

echo ""
echo "========================================"
echo "PTQ Test Complete!"
echo "========================================"
echo "Training log:      ${OUT_DIR}/train_${RUN_NAME}.log"
echo "Calibration log:   ${RUN_DIR}/calibrate_${RUN_NAME}.log"
echo "Quantization log:  ${RUN_DIR}/quant_${RUN_NAME}.log"
echo "Evaluation log:    ${RUN_DIR}/eval_${RUN_NAME}.log"
echo ""
echo "Final accuracy:"
grep 'Top1:' ${RUN_DIR}/eval_${RUN_NAME}.log | tail -1
