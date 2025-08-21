#!/bin/sh
python train_Now.py --epochs 10 --batch-size 256 \
  --optimizer Adam --lr 0.001 --weight-decay 0.0002 \
  --use-bias --deterministic \
  --model ai85indoorenvnetv1 --dataset IndoorEnvironmentAug --data data/indoor_environment \
  --compress policies/schedule-indoor-env.yaml \
  --qat-policy policies/qat_policy_indoor.yaml \
  --tensorboard \
  --device MAX78002 --name indoor_run_8Bit "$@"