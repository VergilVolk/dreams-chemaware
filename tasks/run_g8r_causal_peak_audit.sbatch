#!/bin/bash
#SBATCH --job-name=g8r_causal_peaks
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=18:00:00
#SBATCH --output=data/validation/g8r_causal_peaks_%j.out
#SBATCH --error=data/validation/g8r_causal_peaks_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "=== causal peak audit started: $(date) ==="
python -u tasks/run_g8r_causal_peak_audit.py \
  --device cuda \
  --batch-size 128 \
  --random-repeats 3 \
  --protected-controls 2000 \
  --bootstrap 2000 \
  --seed 20260824
echo "=== causal peak audit completed: $(date) ==="
