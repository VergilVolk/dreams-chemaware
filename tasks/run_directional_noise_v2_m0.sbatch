#!/bin/bash
#SBATCH --job-name=dir_noise_v2_m0
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=data/validation/dir_noise_v2_m0_%j.out
#SBATCH --error=data/validation/dir_noise_v2_m0_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "=== directional-noise v2 M0 started: $(date) ==="
python -u tasks/audit_directional_noise_v2_pool.py \
  --minimum-spectra 3 \
  --maximum-spectra 12 \
  --fragment-tolerance 0.02 \
  --core-prevalence 0.60 \
  --conditional-prevalence 0.40 \
  --conditional-intensity-max 0.20 \
  --ppm 10.0 \
  --seed 20260824
echo "=== directional-noise v2 M0 completed: $(date) ==="
