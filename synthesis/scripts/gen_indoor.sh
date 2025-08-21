#!/bin/sh
DEVICE="MAX78002"
TARGET="../indoor_env"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

python ai8xize.py --test-dir $TARGET --prefix indoor_env --checkpoint-file ../ai8x-training/logs/indoor_run_8Bit___2025.08.16-143727/indoor_run_8Bit_qat_best_q8.pth.tar --config-file networks/indoorenvnet-v1-hwc.yaml --sample-input tests/sample_indoorenvironment.npy $COMMON_ARGS --overwrite  --softmax --compact-data --mexpress  "$@"