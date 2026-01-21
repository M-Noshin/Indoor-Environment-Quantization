#!/bin/bash
# PTQ Mixed-Precision Sweep
# Takes a trained float checkpoint and sweeps over multiple quantization configs
# Flow: Calibrate → Quantize → Evaluate (for each config)

set -e  # Exit on error

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate max

# Setup Python path for distiller
export PYTHONPATH=$PWD/distiller:${PYTHONPATH:-}

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input checkpoint (already trained)
CHECKPOINT="${1:-ptq_test_output/indoor_ptq_test_seed_42_L101___2026.01.21-210016/indoor_ptq_test_seed_42_L101_best.pth.tar}"

if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT}"
    echo "Usage: $0 <path_to_checkpoint.pth.tar>"
    exit 1
fi

# Model and dataset settings
SEED=42
INPUT_LENGTH=101
BATCH_SIZE=256
WEIGHT_DECAY=0.0005
DEVICE="MAX78002"
DATA_DIR="data/indoor_environment"
OUT_DIR="ptq_mixed_sweep_output"
CALIB_SPLIT="trainval"  # train | trainval

# Define configs to sweep (add/remove as needed)
# Format: "conv1_bits conv2_bits fc1_bits fc2_bits"
CONFIGS=(
    "8 8 8 8"     # Baseline INT8
    "4 4 4 4"     # All 4-bit
    "2 2 2 2"     # All 2-bit
    "4 4 2 8"     # Mixed: conv=4, fc1=2, fc2=8
    "8 4 2 8"     # Mixed: conv1=8, conv2=4, fc1=2, fc2=8
    "4 4 4 8"     # Mixed: conv=4, fc=4/8
    "8 8 2 8"     # Mixed: conv=8, fc1=2, fc2=8
    "2 4 8 8"     # Mixed: conv1=2, conv2=4, fc=8
)

echo "========================================"
echo "PTQ Mixed-Precision Sweep"
echo "========================================"
echo "Checkpoint: ${CHECKPOINT}"
echo "Configs: ${#CONFIGS[@]}"
echo "Device: ${DEVICE}"
echo "Calibration split: ${CALIB_SPLIT}"
echo "========================================"
echo ""

mkdir -p "${OUT_DIR}"

# Results tracking
RESULTS_FILE="${OUT_DIR}/sweep_results.csv"
echo "config,conv1_bits,conv2_bits,fc1_bits,fc2_bits,test_accuracy,calibration_time,quantization_time,evaluation_time" > "${RESULTS_FILE}"

# ============================================================================
# SWEEP LOOP
# ============================================================================

for idx in "${!CONFIGS[@]}"; do
    CONFIG=(${CONFIGS[$idx]})
    CONV1_BITS=${CONFIG[0]}
    CONV2_BITS=${CONFIG[1]}
    FC1_BITS=${CONFIG[2]}
    FC2_BITS=${CONFIG[3]}
    CONFIG_NAME="INT_${CONV1_BITS}-${CONV2_BITS}-${FC1_BITS}-${FC2_BITS}"
    
    echo ""
    echo "========================================"
    echo "Config $((idx+1))/${#CONFIGS[@]}: ${CONFIG_NAME}"
    echo "========================================"
    
    # Create QAT policy
    QAT_POLICY_FILE="${OUT_DIR}/qat_policy_${CONFIG_NAME}.yaml"
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
    
    # Output paths
    RUN_DIR="${OUT_DIR}/${CONFIG_NAME}"
    mkdir -p "${RUN_DIR}"
    CALIBRATED_CKPT="${RUN_DIR}/calibrated.pth.tar"
    QUANT_CKPT="${RUN_DIR}/calibrated_q.pth.tar"
    
    # Step 1: Calibrate
    echo "Step 1/3: Calibrating..."
    CALIB_START=$(date +%s)
    python calibrate_ptq_simple.py \
      --checkpoint "${CHECKPOINT}" \
      --output "${CALIBRATED_CKPT}" \
      --model ai85indoorenvnetv2 \
      --dataset IndoorEnvironment_1D \
      --data "${DATA_DIR}" \
      --input-1d-length "${INPUT_LENGTH}" \
      --device "${DEVICE}" \
      --batch-size "${BATCH_SIZE}" \
      --z-score 2.0 \
      --calib-split "${CALIB_SPLIT}" \
      --qat-policy "${QAT_POLICY_FILE}" \
      2>&1 | tee "${RUN_DIR}/calibrate.log"
    CALIB_END=$(date +%s)
    CALIB_TIME=$((CALIB_END - CALIB_START))
    
    if [ ! -f "${CALIBRATED_CKPT}" ]; then
        echo "ERROR: Calibration failed for ${CONFIG_NAME}"
        echo "${CONFIG_NAME},${CONV1_BITS},${CONV2_BITS},${FC1_BITS},${FC2_BITS},ERROR,${CALIB_TIME},0,0" >> "${RESULTS_FILE}"
        continue
    fi
    
    # Step 2: Quantize
    echo "Step 2/3: Quantizing..."
    QUANT_START=$(date +%s)
    python ../ai8x-synthesis/quantize.py \
      "${CALIBRATED_CKPT}" \
      "${QUANT_CKPT}" \
      --device "${DEVICE}" \
      -v \
      2>&1 | tee "${RUN_DIR}/quantize.log"
    QUANT_END=$(date +%s)
    QUANT_TIME=$((QUANT_END - QUANT_START))
    
    if [ ! -f "${QUANT_CKPT}" ]; then
        echo "ERROR: Quantization failed for ${CONFIG_NAME}"
        echo "${CONFIG_NAME},${CONV1_BITS},${CONV2_BITS},${FC1_BITS},${FC2_BITS},ERROR,${CALIB_TIME},${QUANT_TIME},0" >> "${RESULTS_FILE}"
        continue
    fi
    
    # Step 3: Evaluate
    echo "Step 3/3: Evaluating..."
    EVAL_START=$(date +%s)
    python train.py \
      --deterministic \
      --optimizer Adam \
      --model ai85indoorenvnetv2 \
      --dataset IndoorEnvironment_1D \
      --data "${DATA_DIR}" \
      --input-1d-length "${INPUT_LENGTH}" \
      --device "${DEVICE}" \
      --qat-policy "${QAT_POLICY_FILE}" \
      --use-bias \
      --weight-decay "${WEIGHT_DECAY}" \
      --evaluate \
      --exp-load-weights-from "${QUANT_CKPT}" \
      -8 \
      --confusion \
      --print-freq 10 \
      --compiler-mode none \
      --out-dir "${OUT_DIR}" \
      --seed "${SEED}" \
      --name "${CONFIG_NAME}_eval" \
      2>&1 | tee "${RUN_DIR}/evaluate.log"
    EVAL_END=$(date +%s)
    EVAL_TIME=$((EVAL_END - EVAL_START))
    
    # Extract accuracy
    TEST_ACC=$(grep "Top1:" "${RUN_DIR}/evaluate.log" | tail -1 | grep -oP 'Top1:\s+\K[0-9.]+' || echo "ERROR")
    
    echo "${CONFIG_NAME},${CONV1_BITS},${CONV2_BITS},${FC1_BITS},${FC2_BITS},${TEST_ACC},${CALIB_TIME},${QUANT_TIME},${EVAL_TIME}" >> "${RESULTS_FILE}"
    
    echo "✓ ${CONFIG_NAME}: ${TEST_ACC}% (calib=${CALIB_TIME}s, quant=${QUANT_TIME}s, eval=${EVAL_TIME}s)"
done

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "========================================"
echo "PTQ Mixed-Precision Sweep Complete!"
echo "========================================"
echo ""
echo "Results:"
column -t -s',' "${RESULTS_FILE}"
echo ""
echo "Detailed results saved to: ${RESULTS_FILE}"
echo "Individual run logs in: ${OUT_DIR}/"
