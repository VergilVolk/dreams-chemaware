#!/bin/bash
#SBATCH --job-name=noise_v3_g1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/noise_v3_g1_%j.out
#SBATCH --error=data/validation/noise_v3_g1_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"

echo "=== noise v3 candidate-gradient audit started: $(date) ==="
echo "=== stage 1/2: fail-closed preflight ==="
python -u tasks/check_noise_v3_candidate_gradient.py
echo "=== stage 2/2: full input-gradient intervention audit ==="
python -u tasks/audit_noise_v3_candidate_gradient.py \
  --device cuda \
  --gradient-batch-size 32 \
  --encode-batch-size 128 \
  --top-k-negatives 5 \
  --softmax-temperature 0.10 \
  --selection-attenuation 0.50 \
  --attenuations 0.25 0.50 1.0 \
  --control-repeats 3 \
  --bootstrap 2000 \
  --seed 20260824
echo "=== noise v3 candidate-gradient audit completed: $(date) ==="
