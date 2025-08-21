#!/bin/sh

# Usage:
#   K=5 REPEATS=1 bash ai8x-training/scripts/train_indoor_kfold.sh
# Optional env:
#   K            - number of folds (default 5)
#   REPEATS      - number of repeats with different seeds (default 1)
#   EXTRA_ARGS   - extra args passed to train.py
#   DATASET_NAME - dataset name (default IndoorEnvironmentKFold)
#   MODE         - 'sample' for normal KFold, 'group' for GroupKFold (default 'sample')

K=${K:-5}
REPEATS=${REPEATS:-1}
DATASET_NAME=${DATASET_NAME:-IndoorEnvironmentKFold}
MODE=${MODE:-group}

for r in `seq 0 $((REPEATS-1))`; do
  for f in `seq 0 $((K-1))`; do
    INDOOR_KFOLD_K=${K} INDOOR_KFOLD_FOLD=${f} INDOOR_KFOLD_REPEAT_SEED=${r} INDOOR_KFOLD_MODE=${MODE} \
    python train.py --epochs 20 --batch-size 512 \
      --optimizer Adam --lr 0.001 --weight-decay 0.0005 \
      --use-bias --deterministic \
      --model ai85indoorenvnetv1 --dataset ${DATASET_NAME} --data data/indoor_environment \
      --compress policies/schedule-indoor-env.yaml \
      --qat-policy policies/qat_policy_indoor.yaml \
      --device MAX78002 --name indoor_k${K}_${MODE}_r${r}_f${f} ${EXTRA_ARGS}
  done
done



