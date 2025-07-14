#!/bin/sh

python train.py --deterministic --model ai85indoorenvnetv1 --dataset IndoorEnvironment --data data/indoor_environment --device MAX78002 --qat-policy policies/qat_policy_indoor.yaml --use-bias --evaluate --exp-load-weights-from logs/indoor_run_87___2025.07.13-202612/indoor_run_87_qat_best_q8.pth.tar -8 --confusion --print-freq 10 --save-sample 10 "$@"


