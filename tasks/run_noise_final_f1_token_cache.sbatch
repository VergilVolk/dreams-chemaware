#!/bin/bash
#SBATCH --job-name=noise_f1_tokens
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/noise_final_f1_tokens_%j.out
#SBATCH --error=data/validation/noise_final_f1_tokens_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"

echo "=== F1 full contextual peak-token cache started: $(date) ==="
python -u tasks/encode_noise_final_f1_full_tokens.py
echo "=== F1 full contextual peak-token cache completed: $(date) ==="
