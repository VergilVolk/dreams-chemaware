#!/bin/bash
#SBATCH --job-name=dir_noise_v2_m1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=data/validation/dir_noise_v2_m1_%j.out
#SBATCH --error=data/validation/dir_noise_v2_m1_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "=== directional-noise v2 M1 started: $(date) ==="
python -u tasks/run_directional_noise_v2_margin_audit.py \
  --device cuda \
  --batch-size 128 \
  --queries-per-identity 2 \
  --random-repeats 3 \
  --max-target-peaks 12 \
  --max-mask-fraction 0.20 \
  --ppm 10.0 \
  --bootstrap 5000 \
  --seed 20260824
echo "=== directional-noise v2 M1 completed: $(date) ==="
