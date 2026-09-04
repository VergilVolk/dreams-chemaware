#!/bin/bash
#SBATCH --job-name=noise_f0_protocol
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=data/validation/noise_final_f0_protocol_%j.out
#SBATCH --error=data/validation/noise_final_f0_protocol_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"

echo "=== symmetric noise-finetuning F0 protocol audit started: $(date) ==="
python -u tasks/test_noise_final_f0_protocol.py
python -u tasks/audit_noise_final_f0_protocol.py
echo "=== symmetric noise-finetuning F0 protocol audit completed: $(date) ==="
