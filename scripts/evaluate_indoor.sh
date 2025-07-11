#!/bin/sh

python train.py --deterministic --model ai85indoorenvnetv1 --dataset IndoorEnvironment --data data/indoor_environment --device MAX78000 --qat-policy policies/qat_policy_indoor.yaml --use-bias --evaluate --exp-load-weights-from logs/indoor_run___2025.07.11-175030/indoor_run_qat_best.pth_q8.tar -8 --confusion --print-freq 10 "$@"
