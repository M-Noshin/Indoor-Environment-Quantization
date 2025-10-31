#!/bin/sh
DEVICE="MAX78002"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

# Set input length - change this value to generate for different lengths
LENGTH=51

# Configuration bits (e.g., 2-2-2-2 or 8-4-2-2). Allow override via CONFIG env var.
CONFIG_DEFAULT="8-8-2-4"
: "${CONFIG:=$CONFIG_DEFAULT}"

# Derived forms of CONFIG for different filename conventions
CONFIG_DASH="$CONFIG"                                  # e.g., 8-4-2-2
CONFIG_UNDERSCORE=$(echo "$CONFIG_DASH" | tr '-' '_')  # e.g., 8_4_2_2
CONFIG_COMPACT=$(echo "$CONFIG_DASH" | tr -d '-')      # e.g., 8422

# Determine quantization suffix (q8, q4, q2, or qmixed) unless overridden
if [ -z "$QSUFFIX" ]; then
  IFS='-' read -r Q0 Q1 Q2 Q3 <<EOF
$CONFIG_DASH
EOF
  if [ "$Q0" = "$Q1" ] && [ "$Q1" = "$Q2" ] && [ "$Q2" = "$Q3" ]; then
    QSUFFIX="q${Q0}"
  else
    QSUFFIX="qmixed"
  fi
fi

# Allow caller to override output dir and prefix via env vars
# Defaults
TARGET_DEFAULT="../ai8x-training/ai8x_seed_runs_out/HW_Evaluation/"
PREFIX_DEFAULT="indoor_env_1d_${LENGTH}_q${CONFIG_COMPACT}_MAX"

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
# You can swap to other variants if desired; this uses the qmixed suffix by default
: "${CHECKPOINT:=../ai8x-training/ai8x_seed_runs_out/HW_Evaluation/indoor_mixed_seed_46__L${LENGTH}__${CONFIG_UNDERSCORE}_indoor_mixed_seed_46__L${LENGTH}__${CONFIG_UNDERSCORE}_qat_best_${QSUFFIX}.pth.tar}"

# Ensure output directory exists
mkdir -p "$TARGET"

python ai8xize.py \
  --test-dir "$TARGET" \
  --prefix "$PREFIX" \
  --checkpoint-file "$CHECKPOINT" \
  --config-file networks/indoorenvnet-v2-chw-${LENGTH}.yaml \
  --sample-input tests/sample_indoorenvironment_1d_${LENGTH}.npy \
  $COMMON_ARGS --overwrite --softmax --compact-data --mexpress --max-speed --energy "$@"