#!/bin/sh
python train.py --epochs 10 --batch-size 256 \
  --optimizer Adam --lr 0.001 --weight-decay 0 \
  --use-bias --deterministic \
  --model ai85indoorenvnetv1 --dataset IndoorEnvironment --data data/indoor_environment \
  --compress policies/schedule-indoor-env-phase1-exact.yaml \
  --qat-policy None \
  --device MAX78002 --name indoor_run_no_qat "$@"
  # --qat-policy policies/qat_policy_indoor.yaml \