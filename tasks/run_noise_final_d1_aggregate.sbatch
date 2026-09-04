#!/bin/bash
#SBATCH --job-name=noise_d1_agg
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=data/validation/noise_final_d1_aggregate_%j.out
#SBATCH --error=data/validation/noise_final_d1_aggregate_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"

ROOT_DIR="${D1_ROOT:-data/validation/g8r_noise_final_d1b_adapter}"
python -u tasks/aggregate_noise_final_d1.py --root "$ROOT_DIR"
echo "=== final noise D1 aggregate completed: $(date) ==="
