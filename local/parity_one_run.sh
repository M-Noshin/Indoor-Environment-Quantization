#!/usr/bin/env bash
# Single (L, config, seed) parity: mixed-style vs hawq-style CLIs with unified
# cwd, PYTHONPATH, data, quantize.py, and interpreter. See parity_one_run.py.
#
# Usage:
#   ./parity_one_run.sh
#   ./parity_one_run.sh -- --in-len 91 --config "INT 8-8-2-2" --seed 42 --epochs 2
#
# Same workload on Slurm (1 GPU, max_env, log: local/slurm_parity_<jobid>.out):
#   sbatch /shared/b00090279/Indoor-Environment-Quantization/local/slurm_parity_one_run.sh
#   sbatch .../slurm_parity_one_run.sh --epochs 2 --in-len 91
#
# Optional env overrides:
#   AI8X_TRAINING_ROOT  (default: /shared/b00090279/testMax/ai8x-training)
#   AI8X_SYNTHESIS_ROOT (default: /shared/b00090279/testMax/ai8x-synthesis)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AI8X_TRAINING_ROOT="${AI8X_TRAINING_ROOT:-/Users/hamza/Desktop/testMax/ai8x-training}"
export AI8X_SYNTHESIS_ROOT="${AI8X_SYNTHESIS_ROOT:-/Users/hamza/Desktop/testMax/ai8x-synthesis}"

exec python3 "${SCRIPT_DIR}/parity_one_run.py" "$@"
