#!/bin/sh
DEVICE="MAX78002"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

# Set input length - change this value to generate for different lengths
LENGTH=101

# Allow caller to override output dir and prefix via env vars
# Defaults
TARGET_DEFAULT="../ai8x-training/ai8x_seed_runs_out/HW_Evaluation/"
PREFIX_DEFAULT="indoor_env_1d_${LENGTH}"

# Support aliases for output directory
if [ -n "$OUT_DIR" ]; then
  TARGET="$OUT_DIR"
fi
if [ -n "$OUTPUT_DIR" ]; then
  TARGET="$OUTPUT_DIR"
fi

# Apply defaults if not provided
: "${TARGET:=$TARGET_DEFAULT}"
: "${PREFIX:=$PREFIX_DEFAULT}"

# Default checkpoint path - override with CHECKPOINT env var
: "${CHECKPOINT:=../ai8x-training/ai8x_seed_runs_out/HW_Evaluation/indoor_mixed_seed_46__L${LENGTH}__8_8_8_8_indoor_mixed_seed_46__L${LENGTH}__8_8_8_8_qat_best_q8.pth.tar}"

# Ensure output directory exists
mkdir -p "$TARGET"

python ai8xize.py \
  --test-dir "$TARGET" \
  --prefix "$PREFIX" \
  --checkpoint-file "$CHECKPOINT" \
  --config-file networks/indoorenvnet-v2-chw-${LENGTH}.yaml \
  --sample-input tests/sample_indoorenvironment_1d_${LENGTH}.npy \
  $COMMON_ARGS --overwrite --softmax --compact-data --mexpress "$@"