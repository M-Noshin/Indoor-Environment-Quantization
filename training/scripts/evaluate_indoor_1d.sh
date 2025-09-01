#!/bin/sh

python train.py --deterministic --model ai85indoorenvnetv2 --dataset IndoorEnvironment_1D --data data/indoor_environment --device MAX78002 --qat-policy policies/qat_policy_indoor.yaml --use-bias --deterministic --weight-decay 0.0005 --evaluate --exp-load-weights-from ai8x_seed_runs_out/indoor_run_1D_seed_42___2025.09.01-154918/indoor_run_1D_seed_42_qat_best_q8.pth.tar -8 --confusion --print-freq 10 --save-sample 10 --out-dir ai8x_seed_runs_out "$@"
