#!/bin/bash
#SBATCH --job-name=dir_noise_m1b
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/dir_noise_m1b_%j.out
#SBATCH --error=data/validation/dir_noise_m1b_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "=== directional-noise v2 M1b started: $(date) ==="
python -u tasks/run_directional_noise_v2_candidate_headroom.py \
  --device cuda \
  --batch-size 128 \
  --random-repeats 10 \
  --bootstrap 5000 \
  --seed 20260824
echo "=== directional-noise v2 M1b completed: $(date) ==="
