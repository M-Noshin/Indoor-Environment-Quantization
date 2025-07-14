#!/bin/sh
python train.py --epochs 20 --batch-size 256 \
  --optimizer Adam --lr 0.001 --weight-decay 0.0002 \
  --use-bias --deterministic \
  --model ai85indoorenvnetv1 --dataset IndoorEnvironment --data data/indoor_environment \
  --compress policies/schedule-indoor-env.yaml \
  --qat-policy policies/qat_policy_indoor.yaml \
  --device MAX78002 --name indoor_run_87 "$@"