#!/bin/sh
DEVICE="MAX78002"
TARGET="../indoor_env"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

# Use latest best q8 checkpoint from ai8x_seed_runs_out/checkpoints unless CHECKPOINT is provided
CHECKPOINT_GLOB="../ai8x-training/ai8x_seed_runs_out/**/*_qat_best_q8.pth.tar"
if [ -z "$CHECKPOINT" ]; then
  # Find the most recent *_qat_best_q8.pth.tar anywhere under ai8x_seed_runs_out
  CHECKPOINT="$(ls -t ${CHECKPOINT_GLOB} 2>/dev/null | head -n 1)"
fi

if [ ! -f "$CHECKPOINT" ]; then
  echo "❌ No checkpoint found. Set CHECKPOINT env var or ensure files matching ${CHECKPOINT_GLOB} exist."
  exit 1
fi

python ai8xize.py \
  --test-dir $TARGET \
  --prefix indoor_env_1d \
  --checkpoint-file "../ai8x-training/ai8x_seed_runs_out/indoor_run_1D_seed_42___2025.09.01-154918/indoor_run_1D_seed_42_qat_best_q8.pth.tar" \
  --config-file networks/indoorenvnet-v2-chw.yaml \
  --sample-input tests/sample_indoorenvironment_1d.npy \
  $COMMON_ARGS --overwrite --softmax --compact-data --mexpress "$@"



